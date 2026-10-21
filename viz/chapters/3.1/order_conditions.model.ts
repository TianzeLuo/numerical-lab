/**
 * `order_conditions`（3.1 章 viz 3）的 model：每棵有根树一条方程，残差 = |1 − γΦ|。
 *
 * 面板与图层（顺序即绘制顺序）
 * ----------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `trees` | 单位框边框（0…1，等比例：树形本身是结论的一部分） |
 * | 2 | line | `trees` | 每棵树的边（按 `edge_tree` 分段，段间不连线——不许出现横穿画面的假边） |
 * | 3 | line | `trees` | 每棵树的节点（圆点） |
 * | 4 | line | `residual` | 残差地板 1e-17 参考线（对数轴下沿：那里是"算不出来"，不是"很小"） |
 * | 5 | line | `residual` | 每条条件的残差 stem：**满足=蓝、不满足=红**，扫到当前条为止 |
 * | 6 | line | `residual` | 每条条件的 γ 标记（叠加在残差上，一眼看到"密度"那一列） |
 *
 * 时间轴语义（manifest `time.mode="scrub"`，`tMax=17`）
 * ---------------------------------------------------
 * 滑块是**阶条件序号**：`k` = 已点亮条数（0-based，帧 `meta.cond_index`）。
 * 横轴固定 [0, 17]（5 阶树共 17 条），`max_order` 变小时右边空着——**不裁窗口**，
 * 否则"第 3 条破"与"第 5 条破"在图上会看起来一样（pitfalls 第 2 条）。
 *
 * 三条最容易读反的地方（图上怎么防）
 * ----------------------------------
 * 1. **地板 1e-17**：满足与不满足之间隔着 16 个数量级，图上看不见中间地带；
 *    到了地板不是"残差小"，是"双精度算不出更小"（面板文字里点名）。
 * 2. **树间无强弱**：残差大小与"第几条"无关，枚举顺序不是重要性顺序。
 * 3. **b_hat 从第一棵就不满足**：那不是 bug，是嵌入对的定义（低阶权本来就只有 p̂ 阶）；
 *    非嵌入对方法勾了 `use_hat` 时后端**显式回退**到 b，`meta.hat_fallback` 为真，
 *    图上有专门的角标写出来——绝不静默。
 */

import { formatNumber } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  metaNumber,
  optionalArray,
  palette,
  placeBadge,
  requireArray,
  requireDrawableLayers,
  requirePitfalls,
  requireSameLength,
  sceneShell,
  stackedPanels,
  stringOf,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  lineLayer,
  makeAxis,
  previewBadge,
  series,
  sourceBadge,
  text,
  type LineSeries,
} from "../../shared/src/scene.js";
import { manifest } from "./manifest.gen.js";
import {
  applyOverflowNote,
  axisTitleOf,
  axisYlabelOf,
  countAtCeiling,
  integerTicks,
  logRange,
  positiveMinMax,
  runsBy,
  stems,
  textBlock,
  textTop,
  unitBudget,
  wrapUnits,
} from "./_panel.js";

const CHAPTER = "3.1";
const VIZ = "order_conditions";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "阶条件：每一棵有根树一条方程";
const TIME_MAX = SPEC?.time.tMax ?? 17;

/** 固定角标槽位（见 `model.ts::BADGE_LAYOUT`）。 */
const SLOT = {
  source: 0,
  axisMode: 1,
  auxAxis: 2,
  status: 3,
  preview: 4,
  overflow: 5,
  critical: 6,
  degraded: 7,
} as const;

export const orderConditionsModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const width = Math.max(320, Math.round(opts.width));
  const height = Math.max(240, Math.round(opts.height));

  // —— 必需 ——
  const nodeX = requireArray(frame, "node_x");
  const nodeY = requireArray(frame, "node_y");
  const nodeTree = requireArray(frame, "node_tree");
  const condIndex = requireArray(frame, "cond_index");
  const condOrder = requireArray(frame, "cond_order");
  const condDensity = requireArray(frame, "cond_density");
  const condPhi = requireArray(frame, "cond_phi");
  const condResidual = requireArray(frame, "cond_residual");
  const condPass = requireArray(frame, "cond_pass");
  requireSameLength(nodeX, nodeY, "帧数组 node_x 与 node_y");
  requireSameLength(nodeX, nodeTree, "帧数组 node_x 与 node_tree");
  requireSameLength(condIndex, condResidual, "帧数组 cond_index 与 cond_residual");
  const pitfalls = requirePitfalls(frame);

  // —— 可选 ——
  const degraded: string[] = [];
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：滑块位置改用请求 t");
  const condNow = metaNumber(frame, "cond_index");
  const nPassed = metaNumber(frame, "n_passed");
  const nFailed = metaNumber(frame, "n_failed");
  const maxOrder = metaNumber(frame, "max_order");
  const declared = metaNumber(frame, "declared_order");
  const measured = metaNumber(frame, "measured_order");
  const usingHat = (metaNumber(frame, "using_hat") ?? 0) > 0.5;
  const hatFallback = (metaNumber(frame, "hat_fallback") ?? 0) > 0.5;
  const bHatAvailable = (metaNumber(frame, "b_hat_available") ?? 0) > 0.5;
  const tolUsed = metaNumber(frame, "tol_used");
  const worstResidual = metaNumber(frame, "worst_residual");
  const worstIndex = metaNumber(frame, "worst_index");
  const errFloor = metaNumber(frame, "err_floor");
  const weightNote = stringOf(frame, "weight_note");
  if (weightNote === undefined) degraded.push("帧 strings 缺 weight_note：不声明画的是 b 还是 b_hat（两个解的阶不同）");
  const worstLabel = stringOf(frame, "worst_label");
  const treeLabels = stringOf(frame, "tree_labels");
  if (treeLabels === undefined) degraded.push("帧 strings 缺 tree_labels：树上只标序号与 (ρ, γ)");
  const condLit = optionalArray(frame, "cond_lit");
  if (condLit === undefined) degraded.push("帧缺数组 cond_lit：不区分「已扫到」与「还没扫到」（时间轴的点亮效果失效）");
  const edgeX0 = optionalArray(frame, "edge_x0");
  const edgeY0 = optionalArray(frame, "edge_y0");
  const edgeX1 = optionalArray(frame, "edge_x1");
  const edgeY1 = optionalArray(frame, "edge_y1");
  const edgeTree = optionalArray(frame, "edge_tree");
  const treeCx = optionalArray(frame, "tree_cx");
  const treeCy = optionalArray(frame, "tree_cy");
  if (edgeX0 === undefined || edgeTree === undefined) degraded.push("帧缺数组 edge_*：树上只画节点，不画边");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tMeta ?? opts.t,
    dataSource: stringOf(frame, "data_source") ?? "backend",
    pitfalls,
  });

  // —— 排版：树（正方形单位框）/ 残差 / 条件明细 ——
  // footer 让出 118px：本 viz 的 pitfalls 文案最长（窄画布上它会折到 7 行），
  // 说明块落进 pitfalls 面板就是规则 7 的当场违规。
  // 窄画布上 pitfalls 文案折得更多（面板从 88px 长到 152px），footer 必须跟着宽走。
  const absolute = stackedPanels(width, height, { footer: width >= 712 ? 112 : width >= 600 ? 120 : 170 });
  const nTreeAll = Math.max(1, optionalArray(frame, "tree_cx")?.length ?? 1);
  /** 标签字号固定 10：最宽的标签是 `(4,24,0.042)`（12 个显示单位 = 84px）。 */
  const LABEL_SIZE = 10;
  const LABEL_W = 118;
  const LABEL_H = 13;
  // 本 viz 的两根轴都把 `xlabel` 置空了（轴名写在面板标题里），下方装饰带只剩刻度那一条
  // （`[bottom+3, bottom+15]`，~16px），所以面板间距可以收到 32 而不是通用的 46。
  const GAP = 32;
  const COL_GAP = 54; // 左右两列之间：右列要放自己的 y 刻度标签与轴名
  // 说明块（2 行 = 30px）下面紧接着是树轴的**标题带**（轴上方 18px）——两段都要让出来，
  // 否则说明块的第一行会落在标题带里（校验器报"文本压住坐标轴装饰"）。
  const TEXT_ROWS_H = 40;

  // **左树右残差**：树是正方形（形状本身是结论），塞进横条面板里只会缩成几个像素；
  // 残差是"序号 → 值"的长条，适合竖着放。宽画布一律分栏，窄画布退回上下叠。
  const plotTotal = absolute.bottom - absolute.top;
  const wide = absolute.contentWidth >= 2 * 300 + 24;
  const leftW = wide ? Math.round((absolute.contentWidth - COL_GAP) * 0.54) : absolute.contentWidth;
  const rightW = wide ? absolute.contentWidth - COL_GAP - leftW : absolute.contentWidth;
  // 标签**只在正方形框的横向范围内**排（不能铺满整列）：铺满时最右一列的标签会伸进
  // 右列残差轴的 y 刻度带里（校验器报"文本压住坐标轴装饰"）。所以列数由"可能的最大框边长"定，
  // 再由列数定行数与真实边长——两次定界，收敛得很快。
  const maxSide = Math.max(60, Math.min(leftW - 8, plotTotal - TEXT_ROWS_H - 16));
  const labelCols = Math.max(1, Math.floor(maxSide / LABEL_W));
  const labelRows = Math.max(1, Math.ceil(nTreeAll / labelCols));
  // 高度够不够：说明（2 行）+ 正方形框 + 标签行 + 余量；不够就把说明并进残差面板的说明块。
  const tight = plotTotal - TEXT_ROWS_H - labelRows * LABEL_H - 40 < 120;
  const treeTextH = tight ? 0 : TEXT_ROWS_H;
  const side = Math.max(60, Math.min(leftW - 8, plotTotal - treeTextH - labelRows * LABEL_H - 16));
  const treeH = treeTextH + side + labelRows * LABEL_H + 12;
  const treeRect: [number, number, number, number] = [absolute.left, absolute.top, leftW, treeH];
  const resRect: [number, number, number, number] = wide
    ? [absolute.left + leftW + COL_GAP, absolute.top, rightW, plotTotal]
    : [absolute.left, absolute.top + treeH + GAP, absolute.contentWidth, Math.max(80, plotTotal - treeH - GAP)];
  const treeBudget = unitBudget(leftW - 12);
  const resBudget = unitBudget((wide ? rightW : absolute.contentWidth) - 12);

  const boxX = treeRect[0] + Math.max(4, (leftW - side) / 2);

  const labels = treeLabels !== undefined ? treeLabels.split("|") : [];
  const nTree = treeCx !== undefined ? Math.min(treeCx.length, treeCy?.length ?? 0) : 0;
  const litCount = condLit !== undefined ? countLit(condLit) : Number.POSITIVE_INFINITY;
  // —— 面板文字 ——
  const treeLines: string[] = [];
  // 左列只有 ~58 个单位宽（每个汉字占 2），而且框上面只放得下一行（再往下就是树轴的标题带）：
  // 树的"怎么读"那两句挪到右列的说明块里（那里有 350px 宽）。
  treeLines.push(`框内 = ≤${fmtMaybe(maxOrder)} 阶的全部 ${nTree} 棵有根树。`);
  // 锚在**顶部**：树标签紧跟这段文字往下排（两边都贴底会撞在一起——这是排版的硬约束，
  // 不是"看起来挤"：校验器按 `size+4` 估盒，重叠 >2px 就当场报违规）。
  if (!tight) {
    scene.texts.push(...textBlock(treeRect[0] + 6, textTop(treeRect), treeRect[1] + 20, treeLines, pal.muted, treeBudget, "top"));
  }

  const resLines: string[] = [];
  resLines.push(
    `残差 |1−γΦ|（对数），每棵树一条：蓝=满足（贴地板）、红=不满足；本帧 ${fmtMaybe(nPassed)} 条满足、${fmtMaybe(nFailed)} 条不满足。`,
  );
  resLines.push(`判据 tol_used=${fmtMaybe(tolUsed)}（|1−γΦ| 不超过它就算满足）；满足与不满足之间隔着 16 个数量级，图上只有"贴地板"与"离地板很远"两端。`);
  if (worstResidual !== undefined) {
    resLines.push(
      `最差：${worstLabel !== undefined && worstLabel !== "" ? worstLabel : `#${fmtMaybe(worstIndex)}`}（#${fmtMaybe(worstIndex)}）残差 ${formatNumber(worstResidual, 4)}——残差大不代表「更靠后」。`,
    );
  }

  const k = condNow !== undefined ? Math.round(condNow) : undefined;
  // —— 残差面板的说明块：残差本身 + 当前这一条的明细 + 权重/两条证据链 ——
  // （不再单开一块"明细面板"：960×600 下三块面板的总高放不下，说明块会被截掉一半，
  //  而"哪一条破了 / 画的是 b 还是 b_hat"恰恰是最不能省的两句。）
  const infoLines: string[] = tight ? [...treeLines] : [];
  infoLines.push("树的顺序是**枚举顺序**，不是重要性顺序：第 3 条破与第 5 条破的是同一阶。");
  infoLines.push("标签 ▸n(ρ,γ,Φ)：n=第几棵（第 n 棵 = 第 n−1 条）、ρ=阶、γ=密度、Φ=初等权（要求 Φ=1/γ）。");
  if (k !== undefined && k >= 0 && k < condResidual.length) {
    infoLines.push(
      `当前第 ${k + 1} 条（cond_index=${k}，滑块 meta.t=${fmtMaybe(tMeta)}，tMax=${formatNumber(TIME_MAX, 2)}）：` +
        `ρ=${fmtMaybe(condOrder[k])}、γ=${fmtMaybe(condDensity[k])}、Φ=${fmtMaybe(condPhi[k])}、残差=${fmtMaybe(condResidual[k])}→${(condPass[k] as number) > 0.5 ? "满足" : "不满足"}。`,
    );
  } else {
    infoLines.push(`滑块 meta.t=${fmtMaybe(tMeta)}（tMax=${formatNumber(TIME_MAX, 2)}）：当前条序号超出本帧条件数（${condResidual.length} 条）。`);
  }
  infoLines.push(`权重：${weightNote ?? "帧未声明"}${bHatAvailable ? "" : "（本方法没有 b_hat）"}。`);
  if (labels.length > 0) infoLines.push(`结构串（前几棵）：${clip(labels.slice(0, 8).join(" | "), 70)}`);
  if (measured !== undefined && declared !== undefined) {
    infoLines.push(`树上算出的阶 ${formatNumber(declared, 3)} vs 真跑积分量出的阶 ${formatNumber(measured, 3)}：两条证据链都能读到。`);
  }
  if (hatFallback) infoLines.push("⚠ 勾了 b_hat 但该方法没有：后端显式回退到 b（meta.hat_fallback），图上画的仍是 b。");

  // 残差轴的**文字带**在轴矩形之外（轴下方 28px 的刻度带之后）：说明块不再盖在图上。
  // 行数先自己折一遍（与 `textBlock` 同一把尺子），再把轴的高度让出来。
  const resPieces = [...resLines, ...infoLines].flatMap((body) => wrapUnits(body, resBudget));
  const resTextH = Math.min(resPieces.length * 15 + 6, Math.floor(resRect[3] * 0.5));
  const resAxisRect: [number, number, number, number] = [
    resRect[0],
    resRect[1],
    resRect[2],
    Math.max(56, resRect[3] - resTextH - 28),
  ];

  // 树轴是**正方形**（xlim=ylim=[0,1]，矩形边长 = side）：树的形状本身是结论的一部分，
  // 非等比例的画法会把"链"画成"星"。矩形的宽高由 side 决定，标签在矩形**下方**（像素坐标，
  // 不占数据坐标——数据坐标只留给树）。
  const treeAxisRect: [number, number, number, number] = [boxX, treeRect[1] + treeTextH + 4, side, side];
  const treeAxis = makeAxis("trees", treeAxisRect, {
    xlim: [0, 1],
    ylim: [0, 1],
    xlabel: "",
    ylabel: "",
    title: axisTitleOf(SPEC?.axes, "trees", "有根树（单位框，按阶数枚举）"),
    autoscaled: false,
    xticks: [],
    yticks: [],
    grid: false,
    frame: false,
  });

  // 残差横轴**锁在 [0, tMax=17]**：max_order 变小时右边空着，位置才可比。
  const resAxis = makeAxis("residual", resAxisRect, {
    xlim: [0, TIME_MAX],
    // 窗口要**同时**容得下"贴地板"与"离地板 16 个数量级"：上沿固定到 1e-1（不靠数据撑），
    // 下沿在地板下面一点点。这样"满足"的柱子是一小截、"不满足"的柱子顶到天，两者一眼分得开；
    // 若按数据自适应（全满足时窗口只有两个数量级），16 个数量级的落差就消失了。
    ylim: logRange((errFloor !== undefined && errFloor > 0 ? errFloor : 1e-17) / 3, Math.max(1e-1, positiveMinMax(condResidual)?.[1] ?? 1)),
    yscale: "log",
    xlabel: "",
    ylabel: axisYlabelOf(SPEC?.axes, "residual", "残差"),
    title: axisTitleOf(SPEC?.axes, "residual", "残差 |1−γΦ|（对数，地板 1e-17）"),
    autoscaled: false,
    xticks: integerTicks(0, TIME_MAX, 2),
  });
  scene.axes.push(treeAxis, resAxis);

  // —— 图层 1：单位框（数据坐标 (0,0)–(1,1) 的闭合折线；边长由矩形保证是正方形）——
  scene.layers.push(
    lineLayer("trees", series([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], pal.reference, { width: 1, dash: [4, 3] })),
  );

  // —— 图层 2~3：边（按 edge_tree 分段）与节点 ——
  // 帧里的 node_*/edge_* **已经是归一化数据坐标**（0…1）：直接交给轴去映射。
  // 早先在这里手工换算成像素再当数据画，结果是所有点都落在 xlim 之外（溢出角标当场报出来）。
  if (edgeX0 !== undefined && edgeY0 !== undefined && edgeX1 !== undefined && edgeY1 !== undefined && edgeTree !== undefined) {
    const nEdge = Math.min(edgeX0.length, edgeY0.length, edgeX1.length, edgeY1.length, edgeTree.length);
    const edgeSeries: LineSeries[] = [];
    // 每条边单独成一段：段与段之间**不许**用 NaN 连接（`validateScene` 明确拒绝非有限坐标，
    // 而"这里没有数据"与"坐标是 NaN"在渲染层是两件事——前者不连线，后者会让整条曲线消失）。
    for (const run of runsBy(edgeTree, nEdge)) {
      for (let i = run.start; i < run.end; i += 1) {
        const x0 = edgeX0[i] as number;
        const x1 = edgeX1[i] as number;
        const y0 = edgeY0[i] as number;
        const y1 = edgeY1[i] as number;
        if (![x0, x1, y0, y1].every((v) => Number.isFinite(v))) continue;
        edgeSeries.push(series([x0, x1], [y0, y1], pal.muted, { width: 1.2 }));
      }
    }
    if (edgeSeries.length > 0) scene.layers.push(lineLayer("trees", edgeSeries));
  }
  scene.layers.push(
    lineLayer("trees", [
      series(toArr(nodeX), toArr(nodeY), pal.ink, { width: 1, marker: "o", alpha: 0.85 }),
    ]),
  );

  // —— 树的标签：按**栅格列**排（不是按 tree_cx：树的中心挤在框的两侧，标签会互压）——
  // 一行一棵、单行不折（不设 maxWidth：折行的续行按 `size+4` 占位，行距 22px 装不下两行文字）。
  const labelTop = treeAxisRect[1] + side + 6;
  for (let i = 0; i < nTree; i += 1) {
    const order = condOrder[i];
    const dens = condDensity[i];
    const phi = i < condPhi.length ? condPhi[i] : Number.NaN;
    if (!Number.isFinite(order) || !Number.isFinite(dens)) continue;
    // 单行、≤15 个显示单位（列距 = 框边长 + 18）：右邻是下一列的标签，宽了就会互压。
    // 用共享的 `formatNumber(v, 3)`（匹配合适的字符串长度：12 → "12"、0.0833 → "0.0833" 去掉尾零成 "0.0833"）。
    // 标签数字压到 3 位有效数字（`(4,24,0.042)`）：列距 = 框边长 + 18，宽了就会压住右邻的标签。
    // 面板文字里注明"标签是 3 位有效数字、精确值见残差面板/明细"。
    const body =
      `(${formatNumber(order, 3)},${formatNumber(dens, 3)}` +
      `${Number.isFinite(phi) ? `,${formatNumber(phi, 2)}` : ""})`;
    const lit = i < litCount;
    const col = i % labelCols;
    const row = Math.floor(i / labelCols);
    scene.texts.push(
      text(boxX + col * LABEL_W, labelTop + row * LABEL_H, lit ? `▸${body}` : body, {
        role: "axis",
        size: LABEL_SIZE,
        color: lit ? pal.note : pal.muted,
        align: "left",
      }),
    );
  }
  if (nTree === 0) degraded.push("帧缺 tree_cx/tree_cy：树上不写标签（不按网格反推位置）");

  // —— 图层 4~6：残差地板线、残差 stem、γ 标记 ——
  if (errFloor !== undefined && errFloor > 0) {
    scene.layers.push(
      lineLayer(
        "residual",
        series(resAxis.xlim, [errFloor, errFloor], pal.reference, { width: 1, dash: [2, 4], label: `地板 ${formatNumber(errFloor, 3)}（双精度下限，不是「很小」）` }),
      ),
    );
  }
  const base = resAxis.ylim[0];
  const passStems = stems(condIndex, condResidual, base, (_v, i) => (condPass[i] as number) > 0.5);
  const failStems = stems(condIndex, condResidual, base, (_v, i) => (condPass[i] as number) <= 0.5 && i < litCount);
  const laterStems = stems(condIndex, condResidual, base, (_v, i) => (condPass[i] as number) <= 0.5 && i >= litCount);
  if (passStems.length > 0) {
    scene.layers.push(lineLayer("residual", passStems.map((s, k) => series(s.x, s.y, pal.curves[0], { width: 2, label: k === 0 ? "满足的条件（残差在地板上）" : undefined }))));
  }
  if (failStems.length > 0) {
    scene.layers.push(lineLayer("residual", failStems.map((s, k) => series(s.x, s.y, pal.warn, { width: 2.6, label: k === 0 ? "不满足的条件（残差 ~10^{-1}）" : undefined }))));
  }
  if (laterStems.length > 0) {
    scene.layers.push(lineLayer("residual", laterStems.map((s, k) => series(s.x, s.y, pal.cursor, { width: 1.4, label: k === 0 ? "还没扫到的条件（时间轴未到）" : undefined }))));
  }
  // 密度 γ 只写在树标签与明细里：它落在 [1, 24]，而残差轴是 1e-18…1e-15 的对数轴——
  // 画上去会**跑出轴外**（溢出角标会如实报出来），所以不画。
  if (condNow !== undefined && Number.isFinite(condNow)) {
    scene.layers.push(
      lineLayer("residual", series([condNow, condNow], resAxis.ylim, pal.marker, { width: 1.1, dash: [3, 3] })),
    );
  }


  scene.texts.push(
    ...textBlock(
      resRect[0] + 6,
      resAxisRect[1] + resAxisRect[3] + 28,
      resRect[1] + resRect[3],
      [...resLines, ...infoLines],
      pal.muted,
      resBudget,
      "top",
    ),
  );

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge("拖动时按显示空间插值"), SLOT.preview, width));
  }
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "残差轴：固定对数窗口（地板 1e-17 与 10^{-1} 同屏）；单位框固定，不随 max_order 自适应", 8, 8), SLOT.auxAxis, width),
  );
  const statusBits: string[] = [];
  statusBits.push(
    `${stringOf(frame, "method") ?? "该方法"}　p=${fmtMaybe(declared)}${bHatAvailable ? `　p̂=${fmtMaybe(metaNumber(frame, "order_hat"))}` : ""}　` +
      `${usingHat ? "画 b_hat" : "画 b"}`,
  );
  statusBits.push(
    `条件 ${condResidual.length} 条：满足 ${fmtMaybe(nPassed)} / 不满足 ${fmtMaybe(nFailed)}` +
      (k !== undefined ? `　当前 #${k}` : ""),
  );
  scene.badges.push(placeBadge(badge("status", statusBits.join("　"), 8, 8), SLOT.status, width));
  if (hatFallback) {
    scene.badges.push(
      placeBadge(badge("status", clipBadgeLocal(`⚠ b_hat 回退：${stringOf(frame, "method") ?? "该方法"} 没有低阶权，画的是 b`, width), 8, 8), SLOT.critical, width),
    );
  }

  requireDrawableLayers(scene);
  const rawNonFinite = countNonFinite(nodeX) + countNonFinite(nodeY) + countNonFinite(condResidual) + countNonFinite(condPhi);
  const extras: string[] = [];
  if (errFloor !== undefined) {
    const atFloor = countAtCeiling(condResidual, errFloor);
    if (atFloor > 0) extras.push(`${atFloor} 条残差压在地板 ${formatNumber(errFloor, 3)} 上（= 满足，不代表更小）`);
  }
  if (condPass.length > 0 && nFailed !== undefined && nFailed > 0) extras.push(`${Math.round(nFailed)} 条条件不满足（方法实际只有 ${fmtMaybe(measured)} 阶）`);
  if (usingHat && bHatAvailable) extras.push("画的是 b_hat（低阶权）：残差从第一棵就可能大，这是嵌入对的定义");
  if (hatFallback) extras.push("use_hat 已回退到 b（见 meta.hat_fallback）");
  applyOverflowNote(scene, extras.join("；"), rawNonFinite, { slot: SLOT.overflow, width });

  const deg = degradedBadgeLocal(degraded, SLOT.degraded, width);
  if (deg !== undefined) scene.badges.push(deg);
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

function toArr(a: ArrayLike<number>): number[] {
  const out = new Array<number>(a.length);
  for (let i = 0; i < a.length; i += 1) out[i] = a[i] as number;
  return out;
}

/** 已点亮的条数（`cond_lit` 是 0/1 掩码；非有限一律当"未点亮"）。 */
function countLit(mask: ArrayLike<number>): number {
  let n = 0;
  for (let i = 0; i < mask.length; i += 1) if ((mask[i] as number) > 0.5) n += 1;
  return n;
}

/** 数值的紧凑显示（undefined 显示为「—」而不是 0）。 */
function fmtMaybe(v: number | undefined): string {
  return v === undefined || !Number.isFinite(v) ? "—" : Number.isInteger(v) ? String(v) : formatNumber(v, 4);
}

/** 把长串裁到 `units` 个显示宽度单位（角标与面板都要用同一把尺子）。 */
function clip(body: string, units: number): string {
  if (body.length <= units) return body;
  const parts = body.split(/(\$[^$]*\$|\\\([\s\S]*?\\\))/g).filter((p) => p !== "");
  let out = "";
  let used = 0;
  for (const part of parts) {
    const w = (part.match(/[\u2e80-\uffff]/g)?.length ?? 0) * 2 + part.replace(/[\u2e80-\uffff]/g, "").length;
    if (used + w > Math.max(1, units - 1)) break;
    out += part;
    used += w;
  }
  return `${out.trimEnd()}…`;
}

/** 角标预算裁剪（与 `model.ts::clipBadgeText` 同口径，避免 import 私有工具）。 */
function clipBadgeLocal(body: string, width: number): string {
  const cols = width >= 2 * 344 + 24 ? 2 : 1;
  const usable = cols >= 2 ? 344 : Math.max(60, width - 16);
  const units = Math.max(8, Math.floor((usable - 10) / 7));
  return clip(body, units);
}

/** 降级角标（槽位固定在本章的 SLOT.degraded）。 */
function degradedBadgeLocal(items: readonly string[], slot: number, width: number) {
  if (items.length === 0) return undefined;
  const head = items.slice(0, 2).join("；");
  const rest = items.length > 2 ? `；等 ${items.length} 项` : "";
  return placeBadge(badge("status", `⚠ 降级：${head}${rest}`, 8, 8), slot, width);
}
