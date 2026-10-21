/**
 * `control_points`（2.2 章）的 model：曲线 + 曲率梳 + 节点跳变 + 曲率/能量。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `main` | 样条曲线 S(x)（`x_grid`/`curve`） |
 * | 2 | line | `main` | 曲率梳：每根齿单独一条两点折线（`comb_base`→`comb_tip`） |
 * | 3 | line | `main` | τ=1 时的折线（**只在这一帧真的退化成折线时**画，见下） |
 * | 4 | scatter | `main` | 数据点（`node_x`/`node_y`） |
 * | 5 | scatter | `main` | 探针（`probe`）+ 切向短线段（`probe_tangent`） |
 * | 6 | line | `jump` | `residual`（=|dS'|）与 `jump_d2`，log 纵轴 |
 * | 7 | line | `jump` | 两条判据线：`u/2` 地板、1e-3「肉眼可见折角」 |
 * | 8 | line | `curv` | 曲率 κ(x)（带符号，线性轴） |
 * | 9 | line | `curv` | κ=0 基线 + 探针位置竖线 |
 * | 10 | scatter | `energy` | 样条 / PCHIP / 折线三种能量（log 纵轴；折线恒为 0 → 落在地板上） |
 *
 * 角标槽位（固定，见 `SLOT`）：0 来源 / 1 主轴固定 / 2 节点规模 / 3 跳变轴自适应 /
 * 4 曲率·能量轴自适应 / 5 前端预览（仅 `backend+preview`）/ 6 临界值（τ=1）/ 7 状态与降级。
 *
 * 两条最容易被读反的地方，以及图上怎么防
 * --------------------------------------
 * 1. **「曲率梳切到 local 后哪里更弯」**：`comb` 只决定帧给的是哪一套齿尖
 *    （`comb_tip` 还是 `comb_tip_local`），模型**原样画帧给的那一套**，并把
 *    `strings.comb_name` 与一句结论写进说明带。不这么做的话，"切到 local 之后
 *    到处一样弯"（后端实测 199/201 个非零齿长全等）在图上根本看不出来。
 * 2. **「τ 像平滑旋钮」**：τ 把弯矩乘 (1-τ)。所以跳变面板画的是 `|dS'|`（`residual`）
 *    而不是只看 `|dS''|`——τ=0 时两者都在 u/2 上，分不出"是不是 C²"；而 τ>0 之后
 *    真正被破坏的是 C¹，破绽就写在 `|dS'|` 那条线上，`u/2` 参考线是它的尺子。
 */

import { niceTicks } from "../../shared/src/axes.js";
import {
  applyOverflow,
  canvasSize,
  dataSourceOf,
  degradedBadge,
  metaNumber,
  palette,
  paramNumber,
  paramSummary,
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
  criticalBadge,
  makeAxis,
  scatterLayer,
  sourceBadge,
} from "../../shared/src/scene.js";
import { BC_KEY, BC_NAME, enumName, fixedRange, FLOOR, fmt, linRange, logRange, totalNonFinite } from "./derive.js";
import {
  bandText,
  packColumn,
  combSeries,
  contentBottom,
  finishTexts,
  hLine,
  panelRows,
  panelTop,
  rightColumnFits,
  rightX,
  tooNarrowNote,
  vLine,
} from "./layout.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "2.2";
const VIZ = "control_points";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "控制点、张力与曲率梳";

/** 固定角标槽位（`model.ts::BADGE_LAYOUT`）；长文本角标各占一行的左列，右侧留空。 */
// 8 个槽 = 4 行 × 2 列（`BADGE_LAYOUT.maxSlots`），**不许用 8 号槽**：
// 槽 8 会落到第 5 行（y=106），而主轴标题带是 [104, 120] —— 两者叠在一起。
// 预览那一格并进状态角标（同一行里说得下），把最后一格留给溢出徽标。
const SLOT = {
  source: 0,
  axisMode: 1,
  jumpAxis: 2,
  curvAxis: 3,
  critical: 4,
  status: 5,
  /** 降级说明（规则 6：缺什么就说什么）。 */
  degraded: 6,
  /** 溢出徽标（`applyOverflow`）——与降级分开占槽，否则两者会叠在同一个盒子里。 */
  overflow: 7,
} as const;

/** "肉眼可见的折角"阈值：`|dS'|` 超过它就说明 C¹ 断了（不是舍入）。 */
const KINK_LEVEL = 1e-3;

export const controlPointsModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width } = canvasSize(opts);

  // —— 必需数组：缺一个就抛 ModelInputError（宁可当场炸，也不画半张图）——
  const xGrid = requireArray(frame, "x_grid");
  const curve = requireArray(frame, "curve");
  const slope = requireArray(frame, "slope");
  const curvature = requireArray(frame, "curvature");
  const nodeX = requireArray(frame, "node_x");
  const nodeY = requireArray(frame, "node_y");
  const combBase = requireArray(frame, "comb_base");
  const combTip = requireArray(frame, "comb_tip");
  const combTipLocal = requireArray(frame, "comb_tip_local");
  const jumpNodes = requireArray(frame, "jump_nodes");
  const jumpD1 = requireArray(frame, "jump_d1");
  const jumpD2 = requireArray(frame, "jump_d2");
  const residual = requireArray(frame, "residual");
  const error = requireArray(frame, "error");
  const systemDiag = requireArray(frame, "system_diag");
  const systemRhs = requireArray(frame, "system_rhs");
  const probe = requireArray(frame, "probe");
  const probeTangent = requireArray(frame, "probe_tangent");
  requireSameLength(xGrid, curve, "帧数组 x_grid 与 curve");
  requireSameLength(xGrid, slope, "帧数组 x_grid 与 slope");
  requireSameLength(xGrid, curvature, "帧数组 x_grid 与 curvature");
  requireSameLength(nodeX, nodeY, "帧数组 node_x 与 node_y");
  requireSameLength(jumpNodes, jumpD1, "帧数组 jump_nodes 与 jump_d1");
  requireSameLength(jumpNodes, jumpD2, "帧数组 jump_nodes 与 jump_d2");
  const pitfalls = requirePitfalls(frame);

  // —— 可选的标注信息：缺了就降级（记进 degraded，最后挂成一个 status 角标）——
  const degraded: string[] = [];
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：探针位置改用请求 t（两者可能不同）");
  const tNow = tMeta ?? opts.t;
  const tActual = metaNumber(frame, "t_actual");
  if (tActual === undefined) degraded.push("帧 meta 缺 t_actual：不说明探针被夹到哪个 x");

  const nNodes = metaNumber(frame, "n_nodes");
  const nIntervals = metaNumber(frame, "n_intervals");
  const bc = enumName(stringOf(frame, "bc_name"), metaNumber(frame, "bc_index"), BC_KEY, BC_NAME);
  if (!bc.known) degraded.push("帧 strings/meta 都没给出边界条件：状态角标只写 τ");
  // τ 的取法：**帧优先**（帧是那一帧真正用的值），帧里没有才退回请求参数并标注
  // （拖动过程中帧可能还没跟上；退回请求值比留空更有用，但绝不悄悄当成 0）。
  const tensionFromFrame = metaNumber(frame, "tension");
  const tensionFromParams = paramNumber(opts.params, "tension");
  const tension = tensionFromParams ?? tensionFromFrame;
  const tensionIsRequest = tensionFromParams !== undefined;
  const combIndex = metaNumber(frame, "comb_scale_index");
  const combName = stringOf(frame, "comb_name");
  if (combName === undefined) degraded.push("帧 strings 缺 comb_name：不自行判断梳齿是哪种尺度");

  const energy = metaNumber(frame, "energy");
  const energyPolyline = metaNumber(frame, "energy_polyline");
  const energyPchip = metaNumber(frame, "energy_pchip");
  const energyRatioPchip = metaNumber(frame, "energy_ratio_pchip");
  const curvMax = metaNumber(frame, "curvature_max");
  const curvAtProbe = metaNumber(frame, "curvature_at_probe");
  const jumpD1Max = metaNumber(frame, "jump_d1_max");
  const jumpD2Max = metaNumber(frame, "jump_d2_max");
  const nodalErrorMax = metaNumber(frame, "nodal_error_max");
  const note = stringOf(frame, "note");
  if (note === undefined) degraded.push("帧 strings 缺 note：结论不自行编造");

  // `slope` 不像 `curve` 那样直接上主面板（主轴固定、只有一条曲线要读），
  // 但它有一个非它不可的用途：**在采样网格上**量"一阶导跨过一个采样点跳了多少"。
  // 节点处的解析跳变由帧的 `jump_d1` 给出；τ>0 时折角如果恰好落在两个采样点之间，
  // 解析跳变仍然在（`jump_nodes` 是节点坐标），而这条网格量给出"图上看到的折角"有多大。
  // 两者都是**读帧数组**，不是前端在算导数。
  const slopeJumpMax = maxAdjacentJump(slope);

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tNow,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 版式：三行面板（主图 / 跳变 / 「曲率 | 能量」两列）——
  // 行高、说明文字写在哪儿，全部交给 `layout.ts::panelRows`：它保证所有面板都落在
  // `[panelTop, contentBottom]` 里（后者已经避开陷阱面板），并算出每行说明的竖直上限。
  const rowsLay = panelRows(width, [214, 96, 116], panelTop(width), contentBottom(scene), undefined, 24);
  const mainRect = rowsLay[0]!.rect;
  const jumpRect = rowsLay[1]!.rect;
  const bottomRect = rowsLay[2]!.rect;
  const halfGap = 52;
  const halfW = Math.max(120, Math.round((bottomRect[2] - halfGap) / 2));
  // 版式上的一个硬事实：**最后一行面板底下没有位置写说明**（它下面就是陷阱面板）。
  // 所以曲率/能量那一栏的数字要并进"跳变"那一栏（在图上仍能读到），而不是默默丢掉。
  const curvRect: [number, number, number, number] = [bottomRect[0], bottomRect[1], halfW, bottomRect[3]];
  const energyRect: [number, number, number, number] = [
    bottomRect[0] + halfW + halfGap,
    bottomRect[1],
    Math.max(120, bottomRect[2] - halfW - halfGap),
    bottomRect[3],
  ];

  // —— 主轴：**固定坐标**（规则 1/4）。范围由本帧数据经 1-2-5 取整定出，
  //     不随后端参数漂移：拖动 y_i 时同一量级的位移才是可比的。 ——
  const xLim = fixedRange(toNumbers(xGrid), 0.1, [0, Math.max(1, nodeX.length - 1)]);
  const shownY: number[] = [];
  for (let i = 0; i < curve.length; i += 1) shownY.push(curve[i] as number);
  for (let i = 0; i < nodeY.length; i += 1) shownY.push(nodeY[i] as number);
  for (let i = 0; i < combTip.length; i += 1) shownY.push(combTip[i] as number);
  const yLim = linRange(shownY, [-2, 2], 6);
  const mainAxis = makeAxis("main", mainRect, {
    xlim: xLim,
    ylim: yLim,
    xlabel: "",
    ylabel: axisYlabel("main", "$y$"),
    title: axisTitle("main", "曲线、数据点与曲率梳（固定坐标）"),
    autoscaled: false,
    xticks: niceTicks(xLim[0], xLim[1], Math.max(3, Math.round(mainRect[2] / 100))),
    yticks: niceTicks(yLim[0], yLim[1], 5),
  });

  // —— 跳变轴：log、自适应；两条曲线 + 两条判据线 ——
  const jumpYLim = logRange([...toNumbers(residual), ...toNumbers(jumpD2)], [FLOOR, 1]);
  const jumpAxis = makeAxis("jump", jumpRect, {
    xlim: [xLim[0] - 0.4, xLim[1] + 0.4],
    ylim: jumpYLim,
    yscale: "log",
    xlabel: "",
    ylabel: axisYlabel("jump", "|dS'|, |dS''|"),
    title: axisTitle("jump", "节点处的跳变（log）"),
    autoscaled: true,
    xticks: [],
    yticks: niceTicks(jumpYLim[0], jumpYLim[1], 5, { log: true }),
  });

  // —— 曲率轴：线性、自适应（κ 有正负，对数轴画不出来）——
  const curvLim = linRange(toNumbers(curvature), [-1, 1], 5);
  const curvAxis = makeAxis("curv", curvRect, {
    xlim: xLim,
    ylim: curvLim,
    xlabel: "",
    ylabel: "kappa",  // 与 manifest 的 ylabel 同义；kappa 用 ASCII，位图字体里才有字形
    title: axisTitle("curv", "曲率 kappa"),
    autoscaled: true,
    xticks: [],
    yticks: niceTicks(curvLim[0], curvLim[1], 5),
  });

  // —— 能量轴：log、自适应。折线的能量恰好是 0（帧里如实给 0）→ log 上画不出来，
  //     落到地板 u/2 上画一个点并写明"这不是小，是 0"。 ——
  const energies: number[] = [];
  if (energy !== undefined) energies.push(energy);
  if (energyPchip !== undefined) energies.push(energyPchip);
  if (energyPolyline !== undefined && energyPolyline > 0) energies.push(energyPolyline);
  const energyYLim = logRange(energies, [FLOOR, 1e3]);
  const energyAxis = makeAxis("energy", energyRect, {
    xlim: [-0.6, 3.6],
    ylim: energyYLim,
    yscale: "log",
    // 这一格的 y 轴名**必须留空**：它右对齐写在 `left−56 … left−42`，而那一带正好是
    // 左边曲率面板的地盘（横向重叠 → 规则 8）。能量是什么由本行说明栏交代。
    xlabel: "",
    ylabel: "",
    title: axisTitle("energy", "曲率能量（log）"),
    autoscaled: true,
    xticks: [],
    yticks: niceTicks(energyYLim[0], energyYLim[1], 5, { log: true }),
  });

  scene.axes.push(mainAxis, jumpAxis, curvAxis, energyAxis);

  // —— 图层 1：曲线 ——
  scene.layers.push({
    kind: "line",
    axes: "main",
    series: [{ x: xGrid, y: curve, color: pal.curves[0], width: 2, label: "样条曲线 $S(x)$（帧 curve）" }],
  });

  // —— 图层 2：曲率梳（每根齿一条两点折线；global 时长 ∝ |κ|，local 时等长）——
  const tip = combIndex === 1 ? combTipLocal : combTip;
  const teeth = combSeries(combBase, tip, combIndex === 1 ? pal.warn : pal.muted, 1);
  if (teeth.length > 0) {
    const [first, ...rest] = teeth;
    scene.layers.push({
      kind: "line",
      axes: "main",
      series: [{ ...(first as (typeof teeth)[number]), label: "曲率梳（$comb_tip$）" }, ...rest],
    });
  } else {
    degraded.push("帧 comb_base/comb_tip 没有可画的梳齿：主轴只画曲线");
  }

  // —— 图层 3：τ=1 的折线。判据取自**帧**：τ 到 1 且 |dS'| 真的到了可见量级。
  //     折线本身不是前端算的——只是把帧里的 node_y 用直线连起来（P8：没有任何新物理量）。 ——
  const jumpD1MaxFrame = jumpD1Max ?? -1;
  const isPolyline = tension !== undefined && tension >= 1 && jumpD1MaxFrame > KINK_LEVEL;
  if (isPolyline && nodeX.length >= 2) {
    scene.layers.push({
      kind: "line",
      axes: "main",
      series: [
        {
          x: nodeX,
          y: nodeY,
          color: pal.exact,
          width: 1.4,
          dash: [5, 4],
          label: "折线（$τ=1$）",
        },
      ],
    });
  }

  // —— 图层 4~5：数据点与探针 ——
  scene.layers.push(scatterLayer("main", nodeX, nodeY, { r: 3.6, color: pal.ink, label: "数据点" }));
  if (probe.length >= 2) {
    const px = probe[0] as number;
    const py = probe[1] as number;
    if (Number.isFinite(px) && Number.isFinite(py)) {
      scene.layers.push(
        scatterLayer("main", [px], [py], { r: 4.2, color: pal.marker, label: "探针" }),
      );
      if (probeTangent.length >= 2) {
        const tx = probeTangent[0] as number;
        const ty = probeTangent[1] as number;
        if (Number.isFinite(tx) && Number.isFinite(ty)) {
          // `probe_tangent` 是帧算好的**单位**方向 `(1, S')/‖·‖`（实测默认帧 = (0.332, 0.943)，
          // 而 S'(x₀)=0.332）。屏幕上"一单位 x"与"一单位 y"的像素长度不同，所以要按画面的
          // 数据纵横比把 y 分量换算过去，否则画出来的箭头方向是错的（这是显示几何，不是物理）。
          const yPerX = (yLim[1] - yLim[0]) / Math.max(1e-9, xLim[1] - xLim[0]);
          const len = 0.06 * (xLim[1] - xLim[0]);
          scene.layers.push({
            kind: "line",
            axes: "main",
            series: [
              {
                x: [px - len * tx, px + len * tx],
                y: [py - len * ty * yPerX, py + len * ty * yPerX],
                color: pal.marker,
                width: 1.6,
                label: "切向（$probe_tangent$）",
              },
            ],
          });
        }
      }
    }
  }

  // —— 图层 6~7：跳变（log）+ 两条判据线 ——
  scene.layers.push({
    kind: "line",
    axes: "jump",
    series: [
      {
        x: jumpNodes,
        y: residual,
        color: pal.residual,
        width: 1.6,
        marker: "o",
        label: "$|dS'|$（residual）",
      },
      {
        x: jumpNodes,
        y: jumpD2,
        color: pal.reference,
        width: 1.4,
        dash: [4, 3],
        marker: "s",
        label: "$|dS''|$（$jump_d2$）",
      },
    ],
  });
  hLine(scene, jumpAxis, FLOOR, pal.note, { label: "$u/2$ 地板" });
  hLine(scene, jumpAxis, KINK_LEVEL, pal.warn, { label: "$1e-3$ 折角" });
  if (Number.isFinite(probe[0] as number)) vLine(scene, jumpAxis, probe[0] as number, pal.cursor, { label: undefined });

  // —— 图层 8~9：曲率 ——
  scene.layers.push({
    kind: "line",
    axes: "curv",
    series: [{ x: xGrid, y: curvature, color: pal.curves[1], width: 1.6, label: "$κ$（curvature）" }],
  });
  hLine(scene, curvAxis, 0, pal.reference, { label: "$κ = 0$" });
  if (Number.isFinite(probe[0] as number)) vLine(scene, curvAxis, probe[0] as number, pal.cursor, { label: "探针位置" });

  // —— 图层 10：能量（log）——
  const energyX: number[] = [];
  const energyY: number[] = [];
  const energyColors: string[] = [];
  if (energy !== undefined) {
    energyX.push(0);
    energyY.push(Math.max(energy, FLOOR));
    energyColors.push(pal.curves[0]);
  }
  if (energyPchip !== undefined) {
    energyX.push(1);
    energyY.push(Math.max(energyPchip, FLOOR));
    energyColors.push(pal.curves[1]);
  }
  if (energyPolyline !== undefined) {
    energyX.push(2);
    // 折线能量恰好是 0：log 轴上画不出来。**标在轴底**（而不是标在 `u/2` 上）——
    // 本轴的 ylim 由样条/PCHIP 两条能量定，根本够不到 2^-53；硬塞一个越界点会被
    // `checkOverflow` 记成"数据被轴裁掉"，那是另一回事（这里没有人被裁，是"0 没有对数"）。
    energyY.push(Math.max(energyPolyline, energyYLim[0]));
    energyColors.push(pal.warn);
  }
  if (energyX.length === 0) {
    degraded.push("帧 meta 没有任何能量标量：能量轴画不出东西");
  } else {
    for (let i = 0; i < energyX.length; i += 1) {
      scene.layers.push(
        scatterLayer("energy", [energyX[i] as number], [energyY[i] as number], {
          r: 4.5,
          color: energyColors[i] as string,
          label: i === 0 ? "∫(S'')²：样条 / PCHIP / 折线（帧 meta 标量）" : undefined,
        }),
      );
    }
  }
  hLine(scene, energyAxis, FLOOR, pal.note, { label: "$u/2$" });

  // —— 文本：三条说明带，一行一个 text 层 ——
  // 说明栏：横坐标固定在所有轴的右侧（`TEXT_COL_X`），纵坐标取该行说明带的顶边——
  // 这样每一格文字都紧挨着它描述的那个面板，又永远不落进任何装饰带。
  const wide = rightColumnFits(width);
  const summary = paramSummary(frame, opts, [
    { name: "tension", label: "$τ$", digits: 3 },
    { name: "n_nodes", label: "节点数" },
  ]);
  degraded.push(...summary.degraded);

  const mainLines = [
    `${summary.text}　bc=${bc.known ? bc.text : "?"}　点 ${fmt(nNodes)} / 段 ${fmt(nIntervals)}`,
    `${combName ?? "comb=?"}：${combIndex === 1 ? "齿长全都一样，读不出「哪里更弯」" : "齿长 ∝ |kappa|，可横向比"}`,
  ];

  const kink = jumpD1MaxFrame > KINK_LEVEL;
  const energyZero = energyPolyline !== undefined && energyPolyline <= 0;
  if (energyPolyline === undefined) degraded.push("帧 meta 缺 energy_polyline：不比较折线能量");

  // —— 文字：三块结论按"行"装进三行面板的右栏 ——
  // `packBands` 从最后一块往前装：某块放不下的行**上移**到前一块（行距够就装得下），
  // 而不是被 `finishTexts` 丢掉——被丢的往往是全图最要紧的那句（「折线能量恒 0」）。
  const jumpLines = [
    `|dS'|max ${fmt(jumpD1Max)}　|dS''|max ${fmt(jumpD2Max)}　S' 一步跳 ${fmt(slopeJumpMax)}`,
    kink ? "→ C¹ 已断：节点处是折角，不是舍入" : "→ 两条都挂在地板上：C² 没破",
    `kappa 峰值 ${fmt(curvMax)}　探针处 ${fmt(curvAtProbe, 3)}　节点残差 max ${fmt(nodalErrorMax)}`,
  ];
  if (note !== undefined && !kink) jumpLines[1] = `${jumpLines[1]}（帧 note：${note}）`;
  else if (note === undefined) {
    degraded.push("帧 strings 缺 note：结论不自行编造（跳变栏只报数字）");
    jumpLines[1] = `${jumpLines[1]}（帧未给出 note）`;
  }
  const tailLines = [
    `∫(S'')²：样条 ${fmt(energy)} / PCHIP ${fmt(energyPchip)}（比 ${fmt(energyRatioPchip, 3)}）/ 折线 ${
      energyPolyline === undefined ? "—" : fmt(energyPolyline)
    }`,
  ];
  if (energyZero) {
    // 折线能量恒为 0 这件事**必须在图上说出来**：log 轴上的 0 落在地板线上，看着像"最小"，
    // 而它的 S'' 是节点处的delta、根本不在 H² 里——这是本 viz 最容易读反的一条。
    tailLines.push("折线能量恒 0：它的 S'' 不在 H² 里，别读成「最光滑」");
  }
  if (wide) {
    const lost = packColumn(
      scene,
      width,
      [mainLines, jumpLines, tailLines],
      [pal.muted, kink || energyZero ? pal.warn : pal.muted, energyZero ? pal.warn : pal.muted],
      rowsLay.map((r) => r.rect[1]),
    );
    if (lost > 0) degraded.push(`${lost} 行结论在本画布下放不下（已并入上一栏或省略）`);
  } else {
    bandText(scene, rowsLay[0]!, width, mainLines, pal.muted, 2);
    bandText(scene, rowsLay[1]!, width, [jumpLines[0] as string], kink ? pal.warn : pal.muted, 1);
    bandText(scene, rowsLay[2]!, width, [tailLines[0] as string], energyZero ? pal.warn : pal.muted, 1);
  }
  if (!wide) {
    scene.texts.push(...tooNarrowNote([rightX(width), rowsLay[2]!.limit - 32, 160, 32], pal.warn, width));
  }

  // —— 角标（固定槽位；长文案在说明栏，角标只做索引）——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  if (jumpAxis.yscale === "log") {
    scene.badges.push(placeBadge(badge("autoscale", "副轴：跳变与曲率自适应（主轴固定）", 8, 8), SLOT.jumpAxis, width));
  }
  if (curvAxis.autoscaled || energyAxis.autoscaled) {
    scene.badges.push(placeBadge(badge("autoscale", "曲率/能量轴随帧自适应；log 轴下界 > 0", 8, 8), SLOT.curvAxis, width));
  }
  // 放在最前面：降级角标文案超长时由 `clipBadgeText` 从尾部截断，
  // 排在 `paramSummary` 后面等于没写。最要紧的一句必须最先出现。
  if (tension === undefined) degraded.unshift("帧缺 tension（请求参数也没给）：不画临界角标，不猜临界值");
  if (tension !== undefined && Math.abs(tension - 1) < 1e-9) {
    if (tensionIsRequest) degraded.push("临界角标按请求参数 tension 判定（拖动中最新的值）");
    scene.badges.push(
      placeBadge(
        criticalBadge("tension", tension, 1, { beyond: "above", unit: "（弯矩乘 1−τ）" }),
        SLOT.critical,
        width,
      ),
    );
  }
  const statusText =
    (scene.dataSource.startsWith("backend+preview") ? "拖动预览：按显示空间插值　" : "") +
    `bc=${bc.known ? bc.text : "?"}　τ=${fmt(tension, 3)}${tensionIsRequest ? "（请求值）" : ""}　` +
    `|dS'|max=${fmt(jumpD1Max)}　|dS''|max=${fmt(jumpD2Max)}`;
  scene.badges.push(placeBadge(badge("status", statusText, 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const bad = totalNonFinite([
    xGrid,
    curve,
    slope,
    curvature,
    nodeX,
    nodeY,
    combBase,
    combTip,
    combTipLocal,
    jumpNodes,
    jumpD1,
    jumpD2,
    residual,
    error,
    probe,
    probeTangent,
    systemDiag,
    systemRhs,
  ]);
  applyOverflow(scene, bad, { slot: SLOT.overflow, width });
  // 文字层的两道新规（互相不重叠、不被陷阱面板压住）统一在这里收口。
  finishTexts(scene, width);
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //
/** `ArrayLike<number>` → `number[]`（只为把帧数组交给 `derive.ts` 的取整函数）。 */
function toNumbers(a: ArrayLike<number>): number[] {
  const out: number[] = [];
  for (let i = 0; i < a.length; i += 1) out.push(a[i] as number);
  return out;
}

/** `slope` 的相邻差分最大值：**帧数组上的显示量**，不是前端算的导数。 */
function maxAdjacentJump(a: ArrayLike<number>): number {
  let m = 0;
  for (let i = 1; i < a.length; i += 1) {
    const d = Math.abs((a[i] as number) - (a[i - 1] as number));
    if (Number.isFinite(d) && d > m) m = d;
  }
  return m;
}

/** manifest 里该轴的标题。 */
function axisTitle(id: string, fallback: string): string {
  return SPEC?.axes.find((a) => a.id === id)?.title ?? fallback;
}

/** manifest 里该轴的 ylabel。 */
function axisYlabel(id: string, fallback: string): string {
  return SPEC?.axes.find((a) => a.id === id)?.ylabel ?? fallback;
}
