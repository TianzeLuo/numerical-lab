/**
 * `projection_3d`（1.3 章）的 model：三维投影几何 + 三种解法的正交性缺陷。
 *
 * 面板与图层
 * ----------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `scene` | 列空间平面片的网格（固定等距投影后的线段） |
 * | 2 | line | `scene` | 列空间平面片的轮廓 |
 * | 3~6 | line | `scene` | `A` 的两列 a₁、a₂（从原点出发的箭头） |
 * | 7 | line | `scene` | `b`（虚线箭头） |
 * | 8 | line | `scene` | 动画点 `p_anim`（实线，t=0 时它就是 b） |
 * | 9 | line | `scene` | 动画残差 `r_anim`（t=0 长度为 0，t=1 就是完整的 r） |
 * | 10 | line | `scene` | `p` 处的直角小方块（垂直是量出来的，不是看出来的） |
 * | 11 | line | `orthogonality` | 三种解法的正交性缺陷（地板点用虚线 + 空心标记） |
 *
 * 三条刻意的决定
 * --------------
 * 1. **固定等距投影，视角写在画面上**：三维场景必须用**确定的**投影画到 2D 画布上
 *    （不随机、不随帧变化，否则动画会抖）。这里是正交投影 + 手动正交化的相机基
 *    （相机在 (1,−1.6,0.85) 方向看向原点，屏幕上 = 世界·right、v = 世界·up）。
 *    视角写进面板文本——不写视角的"三维图"读者无法复现。
 * 2. **`scene` 轴固定 ±2.75/±2.35 且等比例**（manifest 声明 `autoscaled: false`）。
 *    这不是随手取的范围：它是"世界立方体 [−1.8,1.8]³ 的投影外框"（投影是线性的，
 *    立方体的像是以 8 个角点投影为顶点的六边形，实测 u 半径 √2·1.8 ≈ 2.546、
 *    v 半径 1.8·(1+1/√6)/√2 ≈ 1.792）。于是**帧里所有落在声明范围内的几何都在画面里**，
 *    不会出现"内容比轴大、被裁掉还看不出来"的情况；超出这个立方体的（理论上不该有）
 *    由 `clipSeg` 裁到边界并记进 overflow。
 * 3. **"解不出来"与"误差很大"分开表达**（PROTOCOLS §1.3.1）：`method_singular[i]=1` 的
 *    路线在帧里**没有有限值**（`AᵀA` 的 LU 主元恰好为 0，后端抛 `ZeroDivisionError`），
 *    数组值只是地板 `defect_floor`。所以那条路线画成**虚线 + 空心点 + 文字说明**，
 *    并统计进 `meta.n_singular_methods`；`meta.defect_selected` / `rel_error_selected` /
 *    `orthogonality_abs` 为 `null` 时**整项省略**（不写 0，也不写"NaN"）。
 */

import { niceLimits } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  ModelInputError,
  canvasSize,
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
  criticalBadge,
  lineLayer,
  makeAxis,
  previewBadge,
  scatterLayer,
  series,
  sourceBadge,
  type LineSeries,
  pitfallsBand,
} from "../../shared/src/scene.js";
import { contentBottom, fmt, joinParts, kv, overflowBadgeFor, textBlock, textFloor, unitBudget } from "./_shared.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "1.3";
const VIZ = "projection_3d";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "三维投影：残差为什么必须垂直于列空间";

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

/** `log10 κ` 的临界值（manifest；`frame.critical.log10_kappa` 缺失时的回落值）。 */
const KAPPA_CRITICAL = 15.95;

// --------------------------------------------------------------------------- //
// 固定相机（等距投影）：世界 → 屏幕（数据坐标）
// --------------------------------------------------------------------------- //
/**
 * 相机方向（**世界**坐标系里从原点指向相机的单位向量）。三个分量都不为 0，
 * 因此三个世界坐标轴在屏幕上互相分开、不会重合——这是"能看出立体"的最低要求。
 */
const CAM_DIR: readonly [number, number, number] = [0.36, -0.62, 0.62];
/** 屏幕水平方向 = 相机右向量（与 `CAM_DIR`、世界 z 轴正交）。 */
const CAM_RIGHT: readonly [number, number, number] = normalize(cross(CAM_DIR, [0, 0, 1]));
/** 屏幕竖直方向 = up = right × dir（右手系；`dir` 已归一，故 up 也归一）。 */
const CAM_UP: readonly [number, number, number] = cross(CAM_RIGHT, CAM_DIR);

/** `scene` 轴的固定范围：世界立方体 [−1.8,1.8]³ 的投影外框（见文件头第 2 条）。 */
const SCENE_MLIM: [number, number] = [-2.75, 2.75];
const SCENE_NLIM: [number, number] = [-2.35, 2.35];

/** 平面片网格的条数（帧自带 11×11 节点；这里抽 5×5 条线，密了会糊成一团）。 */
const PLANE_LINES = 5;

export const projection3dModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需字段：缺一个就抛 ModelInputError（宁可当场炸，也不画半张图）——
  const colA1 = requireArray(frame, "col_a1");
  const colA2 = requireArray(frame, "col_a2");
  const bVec = requireArray(frame, "b_vec");
  const pVec = requireArray(frame, "p_vec");
  const rVec = requireArray(frame, "r_vec");
  const pAnimRaw = requireArray(frame, "p_anim");
  const rAnimRaw = requireArray(frame, "r_anim");
  const planeX = requireArray(frame, "plane_x");
  const planeY = requireArray(frame, "plane_y");
  const planeZ = requireArray(frame, "plane_z");
  const residual = requireArray(frame, "residual");
  const defect = requireArray(frame, "orthogonality_defect");
  const error = requireArray(frame, "error");
  const methodSingular = requireArray(frame, "method_singular");
  const pitfalls = requirePitfalls(frame);

  const vec3 = (name: string, a: ArrayLike<number>): [number, number, number] => {
    if (a.length !== 3) {
      throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 '${name}' 长度 ${a.length} ≠ 3（三维向量必须是 3 元）`);
    }
    const v: [number, number, number] = [a[0] as number, a[1] as number, a[2] as number];
    if (!v.every((t) => Number.isFinite(t))) {
      throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 '${name}' 含非有限分量：[${v.join(", ")}]`);
    }
    return v;
  };
  const a1 = vec3("col_a1", colA1);
  const a2 = vec3("col_a2", colA2);
  const bWorld = vec3("b_vec", bVec);
  const pWorld = vec3("p_vec", pVec);
  const rWorld = vec3("r_vec", rVec);
  const pAnim = vec3("p_anim", pAnimRaw);
  // `r_anim` 是动画里**残差分量**本身（t=1 时就是完整的 r）；它用来给动画残差的箭头定向，
  // 数值位置仍由 p_anim 与 b 决定（p_anim + r_anim ≡ b 是后端的恒等式，模型不重算残差）。
  const rAnim = vec3("r_anim", rAnimRaw);

  const nPlane = Math.min(planeX.length, planeY.length, planeZ.length);
  requireSameLength(planeX, planeY, "帧数组 plane_x 与 plane_y");
  requireSameLength(planeX, planeZ, "帧数组 plane_x 与 plane_z");
  const planeSide = Math.max(2, Math.round(Math.sqrt(nPlane)));
  if (planeSide * planeSide !== nPlane) {
    throw new ModelInputError(
      `帧 ${CHAPTER}/${VIZ} 的平面片不是方形网格：${nPlane} 个节点（期望 k×k，如后端给的 11×11）`,
    );
  }
  if (defect.length !== residual.length || defect.length !== error.length || defect.length !== methodSingular.length) {
    throw new ModelInputError(
      `帧 ${CHAPTER}/${VIZ} 的逐路线数组长度不一致：orthogonality_defect=${defect.length}、` +
        `residual=${residual.length}、error=${error.length}、method_singular=${methodSingular.length}`,
    );
  }
  const nM = defect.length;

  // —— 可选字段：缺了就降级（记进 degraded，最终挂成一个 status 角标）——
  const degraded: string[] = [];
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：scene.t 用请求 t（两者可能不同）");
  const tNow = tMeta ?? opts.t;
  const blend = metaNumber(frame, "t_blend");
  if (blend === undefined) degraded.push("帧 meta 缺 t_blend：动画进度按请求 t 夹到 [0,1] 显示");
  const blendNow = blend ?? Math.min(1, Math.max(0, opts.t));
  // 动画分解的自洽性：后端的口径是 `p_t = p + t·r`、`r_t = (1−t)·r`，于是
  // t=0 时 r_t = 0（只看得见 b）、t=1 时 r_t = r（两个分量完全分开）。
  // 这里只做"帧的动画数组与文档口径是否一致"的检查，**不重算**任何分量。
  const rAnimZero = rAnim.every((v) => v === 0);
  if (blendNow < 1 && rAnimZero && blendNow > 0) {
    degraded.push(`帧动画残差 r_anim 在 t_blend=${fmt(blendNow, 3)} 上已经为 0（按后端口径应为 (1−t)·r）`);
  }
  if (blendNow >= 1 && !rAnimZero) {
    degraded.push("帧动画残差 r_anim 在 t=1 上不为 0：帧的动画分解与文档口径（t=1 时 r_t = r）不一致");
  }
  const kappaTarget = metaNumber(frame, "kappa_target");
  if (kappaTarget === undefined) degraded.push("帧 meta 缺 kappa_target：不显示目标 κ（不按 t 猜）");
  const kappaMeasured = metaNumber(frame, "kappa_measured");
  const sigmaMin = metaNumber(frame, "sigma_min");
  const sigmaMax = metaNumber(frame, "sigma_max");
  const anglePlane = metaNumber(frame, "angle_b_plane_deg");
  const bNorm = metaNumber(frame, "b_norm");
  const residualNorm = metaNumber(frame, "residual_norm");
  const projectionNorm = metaNumber(frame, "projection_norm");
  const defectSelected = metaNumber(frame, "defect_selected");
  const defectMax = metaNumber(frame, "defect_max");
  const defectFloor = metaNumber(frame, "defect_floor");
  const orthAbs = metaNumber(frame, "orthogonality_abs");
  const relErrorSelected = metaNumber(frame, "rel_error_selected");
  const nSingular = metaNumber(frame, "n_singular_methods");
  const selectedSingular = (metaNumber(frame, "selected_singular") ?? 0) > 0.5;
  const frameSingular = (metaNumber(frame, "singular") ?? 0) > 0.5;
  const selectedMethod = stringOf(frame, "selected_method");
  if (selectedMethod === undefined) degraded.push("帧 strings 缺 selected_method：看不出副轴高亮的是哪条路线");
  const status = stringOf(frame, "status");
  const note = stringOf(frame, "note");
  if (note === undefined) degraded.push("帧 strings 缺 note：状态角标不自行编造结论");
  const methodNames = Array.from({ length: nM }, (_, i) => stringOf(frame, `method_${i}`) ?? `路线 ${i}`);

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tNow,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 排版：上面是正方形三维场景（无装饰，投影窗口自己就是尺子），下面左=文字、右=副轴 ——
  // 为什么不是左右两列：主轴是**正方形**（等比例是这张图的全部意义），而副轴要留出
  // 自己的 y 刻度带（left−56..left−4）。左右并排时正方形一缩小，副轴就被挤到只剩
  // 一百多像素——类别名与数值行都放不下。改成"上一下二"之后，正方形拿满可用高度，
  // 副轴拿到整幅宽度的一半，文字行也不再与副轴的装饰带抢位置（规则 8 会当场报）。
  const base = stackedPanels(width, height);
  // 底部面板（渲染层现画的 `scene.pitfalls`）会随文案长短变高：把内容底边一起收上来，
  // 否则副轴与文本的下半截会被它盖住（实测 1.3 的 fan 帧面板顶边只有 442，而 base.bottom=498）。
  base.bottom = contentBottom(frame, width, height, base.bottom);
  const columnGap = 62;
  // 正方形边长：顶满内容高度（左列只放这一块），右列留给副轴 + 文字
  const side = Math.max(170, Math.floor(Math.min(base.bottom - base.top, base.contentWidth * 0.5)));
  const sceneRect: Rect = [base.left, base.top, side, side];
  // 副轴 150px：够放三条横向缺陷行 + 类别名；省下的高度留给下方 13 行文字
  const auxH = Math.max(110, Math.min(150, Math.round(side * 0.4)));
  const auxX = base.left + side + columnGap;
  const auxW = Math.max(160, base.left + base.contentWidth - auxX);
  const auxRect: Rect = [auxX, base.top, auxW, auxH];

  // `scene` 轴：manifest 声明 `autoscaled: false`。**没有装饰带**——三维投影图上的
  // 数字刻度没有物理含义（那是屏幕坐标，不是世界坐标），所以刻度清空、轴名不设，
  // 为的是把"屏幕 u/v"明确标成显示坐标（见下面 info 行），而不是冒充实测坐标轴。
  const sceneAxis = makeAxis("scene", sceneRect, {
    xlim: SCENE_MLIM,
    ylim: SCENE_NLIM,
    xlabel: "",
    ylabel: "",
    title: "",
    autoscaled: false,
    xticks: [],
    yticks: [],
    grid: false,
    frame: true,
  });

  // `orthogonality` 轴（manifest 声明 `autoscaled: true`、`scale: "log"`）：
  // **横向条形**——类别写在 y 轴位置（手写文本层），x 轴是对数尺度的缺陷值。
  // 为什么不做成竖条：类别名（"正规方程/QR/SVD"）没有 `xtickLabels` 可放（AUTHORING §2.6），
  // 而横条的类别名可以挂在每一条左边、落进 y 刻度那条带里（正好是渲染层留给标签的位置）。
  const dPos = finitePositiveMinMax(defect);
  const floorPos = defectFloor !== undefined && defectFloor > 0 ? defectFloor : 1e-18;
  const yLo = Math.min(dPos?.[0] ?? floorPos, floorPos);
  const yHi = Math.max(dPos?.[1] ?? floorPos * 10, floorPos * 10);
  const auxYlim: [number, number] = niceLimits(yLo, yHi, 6, { log: true, pad: 0.06 });
  if (dPos === undefined) degraded.push("orthogonality_defect 里没有 >0 的有限值：副轴只有地板参考线");
  const catLim: [number, number] = [-0.5, nM - 0.5];
  const auxAxis = makeAxis("orthogonality", auxRect, {
    xlim: auxYlim,
    ylim: catLim,
    xscale: "log",
    xlabel: "缺陷 $\\|A^{\\mathrm{T}}\\hat r\\|/(\\|A\\|_F\\|\\hat r\\|)$",
    ylabel: "",
    title: axisTitle("orthogonality", "三种解法的正交性缺陷（对数）"),
    autoscaled: true,
    xticks: undefined,
    yticks: [],
  });
  scene.axes.push(sceneAxis, auxAxis);

  // —— 图层 1~2：列空间平面片 ——
  const toScreen = (p: readonly [number, number, number]): [number, number] => [
    p[0] * CAM_RIGHT[0] + p[1] * CAM_RIGHT[1] + p[2] * CAM_RIGHT[2],
    p[0] * CAM_UP[0] + p[1] * CAM_UP[1] + p[2] * CAM_UP[2],
  ];
  const planeAt = (i: number, j: number): [number, number, number] => {
    const k = j * planeSide + i;
    return [planeX[k] as number, planeY[k] as number, planeZ[k] as number];
  };
  const win: ClipWindow = { u0: SCENE_MLIM[0], u1: SCENE_MLIM[1], v0: SCENE_NLIM[0], v1: SCENE_NLIM[1] };
  let nClippedSeg = 0;
  let nDroppedPoint = 0;
  const pushClipped = (target: LineSeries[], path: [number, number][], color: string, style: Partial<LineSeries> = {}): void => {
    const res = clipPolyline(path, win);
    nClippedSeg += res.clipped;
    for (const piece of res.parts) {
      if (piece.length < 2) continue;
      target.push(
        series(
          Float64Array.from(piece, (q) => q[0]),
          Float64Array.from(piece, (q) => q[1]),
          color,
          piece === res.parts[0] ? style : { ...style, label: undefined },
        ),
      );
    }
  };

  const gridSeries: LineSeries[] = [];
  const step = Math.max(1, Math.floor((planeSide - 1) / (PLANE_LINES - 1)));
  for (let j = 0; j < planeSide; j += step) {
    const path: [number, number][] = [];
    for (let i = 0; i < planeSide; i += 1) path.push(toScreen(planeAt(i, j)));
    pushClipped(gridSeries, path, pal.reference, { width: 0.8, alpha: 0.55 });
  }
  for (let i = 0; i < planeSide; i += step) {
    const path: [number, number][] = [];
    for (let j = 0; j < planeSide; j += 1) path.push(toScreen(planeAt(i, j)));
    pushClipped(gridSeries, path, pal.reference, { width: 0.8, alpha: 0.55 });
  }
  if (gridSeries.length > 0) {
    scene.layers.push(lineLayer("scene", gridSeries));
  } else {
    degraded.push("列空间平面片没有可绘的线段：主场景只画向量");
  }

  // 平面片轮廓（首尾相接，单独一条，避免与网格线抢图例）
  const outline = [
    toScreen(planeAt(0, 0)),
    toScreen(planeAt(planeSide - 1, 0)),
    toScreen(planeAt(planeSide - 1, planeSide - 1)),
    toScreen(planeAt(0, planeSide - 1)),
    toScreen(planeAt(0, 0)),
  ];
  const outlineSeries: LineSeries[] = [];
  pushClipped(outlineSeries, outline, pal.note, { width: 1.2, label: "列空间 $\\operatorname{col}(A)$（$A$ 的两列张成的平面片）" });
  if (outlineSeries.length > 0) scene.layers.push(lineLayer("scene", outlineSeries));

  // —— 图层 3~6：A 的两列 ——
  const origin: [number, number] = toScreen([0, 0, 0]);
  const arrow = (from: [number, number], to: [number, number], color: string, style: Partial<LineSeries>): LineSeries[] => {
    const head = arrowHead(from, to);
    return [
      series([from[0], to[0]], [from[1], to[1]], color, style),
      ...head.map((h) => series([h[0], h[2]], [h[1], h[3]], color, { width: style.width ?? 1.6 })),
    ];
  };
  const colSeries: LineSeries[] = [];
  for (const [vec, label] of [
    [a1, "a₁（A 的第 1 列）"],
    [a2, "a₂（A 的第 2 列）"],
  ] as const) {
    const seg = clipSegment(origin, toScreen(vec), win);
    if (seg === undefined) {
      nClippedSeg += 1;
      nDroppedPoint += 1;
      continue;
    }
    if (seg.clipped) nClippedSeg += 1;
    colSeries.push(...arrow(origin, seg.b, pal.curves[0], { width: 1.8, label }));
  }
  if (colSeries.length > 0) scene.layers.push(lineLayer("scene", colSeries));
  else degraded.push("a₁/a₂ 的投影全部落在窗口外：主场景看不到列向量");

  // —— 图层 7：b ——
  const bScreen = toScreen(bWorld);
  const bSeg = clipSegment(origin, bScreen, win);
  if (bSeg !== undefined) {
    if (bSeg.clipped) nClippedSeg += 1;
    scene.layers.push(lineLayer("scene", arrow(origin, bSeg.b, pal.marker, { width: 2.2, dash: [6, 3], label: "$b$（右端项，单位向量）" })));
  } else {
    nDroppedPoint += 1;
    degraded.push("b 的投影落在窗口外：主场景看不到 b");
  }

  // —— 图层 8~9：动画分解 p_anim + r_anim（p_anim + r_anim ≡ b）——
  const pScreen = toScreen(pAnim);
  const segPB = clipSegment(pScreen, bScreen, win);
  if (segPB !== undefined) {
    if (segPB.clipped) nClippedSeg += 1;
    scene.layers.push(
      lineLayer("scene", [
        ...arrow(pScreen, segPB.b, pal.exact, {
          width: 2,
          label: `$r$ 的分量（动画 $t=${blendNow.toFixed(2)}$；$t=1$ 时它就是完整的 $r$）`,
        }),
        ...arrow(origin, pScreen, pal.curves[1], {
          width: 1.8,
          label: `$p$ 的分量（动画 $t=${blendNow.toFixed(2)}$；$t=0$ 时它就是 $b$）`,
        }),
      ]),
    );
  } else {
    nDroppedPoint += 1;
    degraded.push("动画点 p_anim 跑到窗口外：不画动画分解（t 的语义见面板文本）");
  }

  // —— 图层 10：p 处的直角小方块（用**帧里的** p、r 方向构造，不是眼睛看的）——
  // 三条边：p → p+ε r̂ → p+ε r̂+ε t̂ → p+ε t̂，其中 t̂ 是平面内与 r 垂直的方向。
  const rLen = Math.hypot(rWorld[0], rWorld[1], rWorld[2]);
  const pHat = pWorld.every((v) => v === 0) ? undefined : normalize(pWorld);
  if (rLen > 0 && pHat !== undefined) {
    const rHat = normalize(rWorld);
    // t̂ = p̂ − (p̂·r̂) r̂（平面内的分量；数值上 p̂⊥r̂，这一步只是把它做正）
    const dot = pHat[0] * rHat[0] + pHat[1] * rHat[1] + pHat[2] * rHat[2];
    const tHat = normalize([pHat[0] - dot * rHat[0], pHat[1] - dot * rHat[1], pHat[2] - dot * rHat[2]]);
    const u = tHat.every((v) => Number.isFinite(v)) ? tHat : pHat;
    const eps = 0.14 * Math.min(rLen, Math.hypot(pWorld[0], pWorld[1], pWorld[2]));
    const q = (s: number, t: number): [number, number] =>
      toScreen([
        pWorld[0] + s * rHat[0] + t * u[0],
        pWorld[1] + s * rHat[1] + t * u[1],
        pWorld[2] + s * rHat[2] + t * u[2],
      ]);
    const sq: LineSeries[] = [];
    pushClipped(
      sq,
      [q(0, 0), q(eps, 0), q(eps, eps), q(0, eps), q(0, 0)],
      pal.warn,
      { width: 1.4, label: "$p$ 处的直角标记（$A^{\\mathrm{T}}\\hat r=0$ 的几何含义）" },
    );
    if (sq.length > 0) scene.layers.push(lineLayer("scene", sq));
  } else {
    degraded.push("r 或 p 的长度为 0：没有直角可标（b 落在列空间里时本来就该是这样）");
  }

  // —— 图层 11：副轴上的三种解法 ——
  const marksX: number[] = [];
  const marksY: number[] = [];
  const singularX: number[] = [];
  const singularRow: number[] = [];
  const rowSeries: LineSeries[] = [];
  let nFloor = 0;
  for (let i = 0; i < nM; i += 1) {
    const v = defect[i] as number;
    const isSingular = (methodSingular[i] as number) > 0.5;
    rowSeries.push(series([auxYlim[0], auxYlim[1]], [i, i], pal.reference, { width: 0.7, alpha: 0.35 }));
    if (isSingular) {
      // 这条路线**没有有限解**：数组里是地板，不是实测缺陷。
      nFloor += 1;
      singularX.push(v > 0 ? v : floorPos);
      singularRow.push(i);
      rowSeries.push(
        series([floorPos, auxYlim[1] * 0.35], [i, i], pal.warn, {
          width: 1.2,
          dash: [4, 3],
          label: i === 0 ? "地板 1e-18（占位）——该路线在此没有有限解，不是实测值" : undefined,
        }),
      );
      continue;
    }
    if (Number.isFinite(v) && v > 0) {
      marksX.push(v);
      marksY.push(i);
    }
  }
  if (rowSeries.length > 0) scene.layers.push(lineLayer("orthogonality", rowSeries));
  if (marksX.length > 0) {
    scene.layers.push(
      scatterLayer("orthogonality", marksX, marksY, { r: 3.4, color: pal.curves[0], label: "实测缺陷（实心点）" }),
    );
    // 高亮选中路线：把它的行标成红点（画在实测点之上）。
    if (selectedMethod !== undefined) {
      const order = ["normal", "qr", "svd"];
      const sel = order.indexOf(selectedMethod);
      if (sel >= 0 && sel < nM && !((methodSingular[sel] as number) > 0.5)) {
        scene.layers.push(
          scatterLayer("orthogonality", [defect[sel] as number], [sel], {
            r: 6,
            color: pal.marker,
            alpha: 0.35,
            label: `当前选中路线（${methodNames[sel] ?? selectedMethod}）`,
          }),
        );
      }
    }
  } else {
    degraded.push("三种解法都没有实测缺陷（全部无有限解）：副轴只剩地板参考线");
  }
  if (singularX.length > 0) {
    // **空心点 + 虚线**：占位值与实测值在图上必须长得不一样（PROTOCOLS §1.3.1 第 2 条）。
    scene.layers.push(
      scatterLayer("orthogonality", singularX, singularRow, {
        r: 5,
        color: pal.warn,
        alpha: 0.35,
        label: "该路线没有有限解（空心：不画实心点，避免把地板读成实测）",
      }),
    );
  }
  // 每个方法的类别名：写在**行内、紧贴每一条线之下**。
  //
  // 为什么不是"行内居中"：行只有 60px 高，居中的标签会撞上轴的标题带/轴名带——
  // `axisDecoBands` 是按轴生成的横条（上 `top−18..top−2`、下两条），**任何 x** 都在里面，
  // 所以标签必须在 y 上避开它们（规则 8 会当场报，实测报过两次）。
  // 为什么不是"轴的左外侧"：那正是 y 刻度标签的装饰带（`left−56..left−4`），同样会撞。
  for (let i = 0; i < nM; i += 1) {
    const name = methodNames[i] ?? `路线 ${i}`;
    const rowCy = auxRect[1] + ((nM - 1 - i + 0.5) / nM) * auxRect[3];
    scene.texts.push({
      x: auxRect[0] + 4,
      y: Math.min(rowCy + 13, auxRect[1] + auxRect[3] - 38),
      text: `${name}${(methodSingular[i] as number) > 0.5 ? "（无有限解）" : ""}`,
      size: 10,
      color: (methodSingular[i] as number) > 0.5 ? pal.warn : pal.muted,
      align: "left",
      role: "label",
    });
  }

  // —— 文本：**右列、副轴下方**那一整块（11 行） ——
  // 三块内容按"读图必须知道"排序：几何约定（3 行）→ 本帧参数（3 行）→ 逐路线实测（5 行）。
  // 每行都按 `unitBudget(auxW)` 量过显示宽度（≤52 单位 ≈ 364px）：超宽会被 `wrapUnits`
  // 折成两行，整块溢出成"另有 N 行，画布放不下"——那正好会吃掉"解不出来 ≠ 误差很大"。
  // 随帧变长的 status/note 去状态角标。
  const infoX = auxX;
  const infoBudget = unitBudget(auxW);
  const info: string[] = [];
  info.push(`固定等距投影：相机 (0.36,−0.62,0.62)，正交、无随机。`);
  info.push(`屏幕 u/v 是世界方向投影；窗口固定 ±2.75/±2.35 等比例。`);
  info.push(
    joinParts([
      kv("log10 κ 目标", kappaTarget !== undefined ? Math.log10(Math.max(kappaTarget, Number.MIN_VALUE)) : undefined, 3),
      kv("实测 κ", kappaMeasured, 3),
      kv("σmax", sigmaMax, 3),
      kv("σmin", sigmaMin, 3),
    ]) ?? "帧 meta 缺 κ 信息：这一行不显示（不重算）",
  );
  info.push(
    joinParts([
      anglePlane === undefined ? undefined : `b 与列空间夹角=${fmt(anglePlane, 3)}°`,
      kv("‖b‖", bNorm, 3),
      kv("‖p‖", projectionNorm, 3),
      kv("‖r‖", residualNorm, 3),
    ]) ?? "帧 meta 缺投影范数：这一行不显示（不重算）",
  );
  info.push(
    `时间轴 t=0..1：动画进度（本帧 t=${fmt(blendNow, 3)}；t=0 残差 0、t=1 拆成 p+r），不是物理量。`,
  );
  // 逐路线实测（每条一行；`nM ≤ 3`）：选哪条路线只看这一行的数字，不看颜色。
  for (let i = 0; i < nM; i += 1) {
    const sing = (methodSingular[i] as number) > 0.5;
    const name = methodNames[i] ?? `路线 ${i}`;
    info.push(
      sing
        ? `${name}：**没有有限解**（缺陷数组里是地板 ${fmt(floorPos, 1)}，占位不是测量）。`
        : `${name}：缺陷 ${fmt(defect[i] as number, 3)}、投影误差 ${fmt(error[i] as number, 3)}。`,
    );
  }
  info.push(
    `**「解不出来」≠「误差很大」**：${nFloor}/${nM} 条无有限解` +
      `${frameSingular ? "（帧 meta.singular=true）" : ""}。`,
  );
  info.push(`地板值（${fmt(floorPos, 1)}）是占位，不是测量值；其余路线照给实测。`);
  info.push(
    selectedSingular
      ? `选中=${selectedMethod ?? "?"}：defect_selected / rel_error_selected / orthogonality_abs ` +
        `都是 null——**整项省略，不写 0**。`
      : joinParts([
          selectedMethod === undefined ? undefined : `选中=${selectedMethod}`,
          kv("缺陷", defectSelected, 3),
          kv("投影误差", relErrorSelected, 3),
          kv("‖Aᵀr̂‖", orthAbs, 3),
        ]) ?? "帧 strings/meta 缺选中路线：这一行不显示",
  );
  scene.texts.push(
    ...textBlock(infoX, auxRect[1] + auxRect[3] + 30, textFloor(scene, base.bottom - 6), info, pal.muted, infoBudget, "top"),
  );


  // —— 角标（固定槽位）——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  if (scene.dataSource.startsWith("backend+preview")) {
    scene.badges.push(placeBadge(previewBadge("拖动时按显示空间插值"), SLOT.preview, width));
  }
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge("autoscale", "副轴：自适应（log 缺陷轴按本帧数据取整；自适应会掩盖量级变化）", 8, 8),
      SLOT.auxAxis,
      width,
    ),
  );
  if (kappaTarget !== undefined) {
    const lg = Math.log10(Math.max(kappaTarget, Number.MIN_VALUE));
    scene.badges.push(
      placeBadge(criticalBadge("log10 κ", lg, criticalOf(frame, "log10_kappa", KAPPA_CRITICAL)), SLOT.critical, width),
    );
  }
  const statusBits: string[] = [];
  if (defectMax !== undefined) statusBits.push(`本帧缺陷最大 ${fmt(defectMax, 3)}`);
  if (nSingular !== undefined && nSingular > 0) {
    statusBits.push(`${fmt(nSingular, 0)}/${nM} 条路线无有限解${selectedSingular ? "（含当前选中路线）" : ""}`);
  }
  if (status !== undefined && status !== "") statusBits.push(status);
  if (note !== undefined) statusBits.push(note);
  if (blend !== undefined) statusBits.push(`动画 t=${fmt(blendNow, 3)}`);  scene.badges.push(
    placeBadge(
      badge("status", clipBadgeText(statusBits.length > 0 ? statusBits.join("；") : "帧未给出可判读的状态"), 8, 8),
      SLOT.status,
      width,
    ),
  );
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(colA1) +
    countNonFinite(colA2) +
    countNonFinite(bVec) +
    countNonFinite(pVec) +
    countNonFinite(rVec) +
    countNonFinite(pAnimRaw) +
    countNonFinite(rAnimRaw) +
    countNonFinite(planeX) +
    countNonFinite(planeY) +
    countNonFinite(planeZ) +
    countNonFinite(residual) +
    countNonFinite(defect) +
    countNonFinite(error) +
    countNonFinite(methodSingular);
  const extra: string[] = [];
  if (nClippedSeg > 0 || nDroppedPoint > 0) {
    extra.push(`${nClippedSeg} 段几何裁到画面边界、${nDroppedPoint} 个点在画面外（固定等比例投影的窗口只有 ±2.75/±2.35）`);
  }
  if (nFloor > 0) {
    extra.push(`${nFloor}/${nM} 条路线无有限解：缺陷数组里是地板 ${fmt(floorPos, 1)}（占位，不是实测）`);
  }
  overflowBadgeFor(scene, nBad, extra, { slot: SLOT.overflow, width });
  // 规则 8 的安全网（两侧校验器都查）：
  //  - 文本层之间不许互相重叠（帧里的说明一变长就会多折一行压住下一块）；
  //  - 文本不许被底部 pitfalls 面板压住（面板高度随帧里 pitfalls 的折行数变，模型事先看不见）。
  // 只丢不移：下移会推进轴的装饰带，那是另一条违规。丢弃的行数会写在画布顶部那一行里。
  const band = pitfallsBand(scene);
  scene.texts = resolveTextOverlaps(scene.texts, [8, 24, width - 16, band === null ? height - 8 : band[1] - 4]);
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具（显示几何，不产生任何新的物理量）
// --------------------------------------------------------------------------- //

function axisTitle(id: string, fallback: string): string {
  return SPEC?.axes.find((a) => a.id === id)?.title ?? fallback;
}

function cross(a: readonly [number, number, number], b: readonly [number, number, number]): [number, number, number] {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function normalize(v: readonly [number, number, number]): [number, number, number] {
  const n = Math.hypot(v[0], v[1], v[2]);
  return n > 0 ? [v[0] / n, v[1] / n, v[2] / n] : [0, 0, 0];
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

/** 线段与矩形窗口求交（Liang–Barsky）：返回 `t ∈ [0,1]` 的参数区间。 */
function clipParam(a: readonly [number, number], b: readonly [number, number], w: ClipWindow): [number, number] | undefined {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  let t0 = 0;
  let t1 = 1;
  const edges: [number, number][] = [
    [-dx, a[0] - w.u0],
    [dx, w.u1 - a[0]],
    [-dy, a[1] - w.v0],
    [dy, w.v1 - a[1]],
  ];
  for (const [p, q] of edges) {
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

/** 把一条线段裁进窗口；返回裁剪后的端点（以及是否真的裁过）。整体在外时返回 `undefined`。 */
function clipSegment(
  a: readonly [number, number],
  b: readonly [number, number],
  w: ClipWindow,
): { a: [number, number]; b: [number, number]; clipped: boolean } | undefined {
  const t = clipParam(a, b, w);
  if (t === undefined) return undefined;
  const at = (s: number): [number, number] => [a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s];
  const clipped = t[0] > 0 || t[1] < 1;
  return { a: at(t[0]), b: at(t[1]), clipped };
}

/** 折线逐段裁剪并拼成若干连续段；`clipped` 统计被裁过的线段数。 */
function clipPolyline(path: readonly [number, number][], w: ClipWindow): { parts: [number, number][][]; clipped: number } {
  const parts: [number, number][][] = [];
  let current: [number, number][] = [];
  let clipped = 0;
  for (let i = 0; i + 1 < path.length; i += 1) {
    const a = path[i] as [number, number];
    const b = path[i + 1] as [number, number];
    if (!inside(a, w) || !inside(b, w)) clipped += 1;
    const seg = clipSegment(a, b, w);
    if (seg === undefined) {
      if (current.length > 0) parts.push(current);
      current = [];
      continue;
    }
    const start: [number, number] = [seg.a[0], seg.a[1]];
    const end: [number, number] = [seg.b[0], seg.b[1]];
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

/** 箭头头的两笔（与 1.2 `unit_ball` 同一套几何：头长取线段长的 35%，上限为窗口的 4.5%）。 */
function arrowHead(from: [number, number], to: [number, number]): [number, number, number, number][] {
  const dx = to[0] - from[0];
  const dy = to[1] - from[1];
  const len = Math.hypot(dx, dy);
  if (!(len > 0)) return [];
  const head = Math.min(len * 0.35, (SCENE_MLIM[1] - SCENE_MLIM[0]) * 0.045);
  const ux = dx / len;
  const uy = dy / len;
  const bx = to[0] - head * ux;
  const by = to[1] - head * uy;
  const px = -uy * head * 0.45;
  const py = ux * head * 0.45;
  return [
    [to[0], to[1], bx + px, by + py],
    [to[0], to[1], bx - px, by - py],
  ];
}
