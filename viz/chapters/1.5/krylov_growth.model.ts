/**
 * `krylov_growth`（1.5 章）的 model：Krylov 基逐列扩张 + 正交性/H 两张**实测**矩阵。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `space` | R^n 的坐标轴像（仅 n=3 时画，给投影一个方位参照） |
 * | 2 | line | `space` | Krylov 基向量 v_j（帧 `basis` 的行；从原点出发） |
 * | 3 | line | `space` | 解轨迹 x_k（帧 `iterates`，带圆点） |
 * | 4 | scatter | `space` | 精确解 x*（帧 `x_star`）与当前步 x_k |
 * | 5 | heat | `hess` | log10\|H\|（帧 `hessenberg`；CG 帧是 0×0 → 不画并说明） |
 * | 6 | heat | `orth` | log10\|正交性矩阵\|（帧 `orthogonality`，行列 = 步 i / 基 j） |
 * | 7 | line | `orth` | 舍入底分界线（该线以上的行只剩噪声） |
 * | 8 | line | `resid` | 相对残差（帧 `residual`，对数轴） |
 * | 9 | line | `resid` | √u 舍入底参考线（低于它正交性矩阵不可读） |
 * | 10 | scatter | `resid` | 当前步残差（帧 `residual_mark`） |
 *
 * 角标槽位（固定，见 `SLOT`）：0 来源 / 1 主轴自适应 / 2 矩阵轴固定 / 3 残差轴自适应 /
 * 4 状态 / 5 溢出 / 6 降级。
 *
 * 几条刻意的决定
 * --------------
 * 1. **两种正交性不能混**：帧的 `strings.orthogonality_name` 明确写出这一帧画的是哪一条
 *    （CG 是 `|r_i^Tr_j|`，GMRES 是 `|r_i^TAv_j|`）。标题与图例一律用它，绝不写成笼统的
 *    「正交性」——把 GMRES 的 `r_i^Tr_j` 当成「应该为 0」去读会得出「实现错了」的错误结论。
 * 2. **矩阵热力图的地板是「没有新方向」**：元素是归一化内积，0 与机器零都压到 1e-16，
 *    色标取 log10 后 0 与 1e-16 同色——这不是「没有数据」（对数**色标**会把 0 判成缺数据涂灰，
 *    那是另一回事），而是「这一步没有新方向」。帧 meta 的 `orthogonality_max` 才是
 *    「屏蔽舍入底以下几行之后的下三角最大值」，写在面板文本里。
 * 3. **残差落到舍入底以下的行要点名**：`√u·‖r_0‖ ≈ 1.5e-8` 之后 `r_i` 只是噪声，
 *    归一化内积会跳到 O(1)（本章 pitfalls 第 7 条实测能到 0.2）。模型用帧里的 `residual`
 *    数组（不是重算的物理量）找出**第一行低于底**的位置，画一条分界线并写明「线以上是噪声」。
 * 4. **H 与正交性矩阵都是「整段算完」的**：后端一次算完整个 Krylov 过程，两张矩阵在所有帧里
 *    逐位相同，游标裁剪不了它们（`hessenberg`/`orthogonality` 没有「前缀」变体）。
 *    这一点写在文本里，免得被读成「第 0 步就已经把整个子空间建好了」。
 * 5. **CG 没有 H 是正常的**：`hessenberg` 在 CG 帧里是 0×0，`heat` 图层又要求至少 2×2，
 *    所以这一帧不画 H 面板（角标里声明），把 `β_j ≡ 0` 的读法留在文本里——
 *    次对角元 β 变 0 才等于「子空间已不变（happy breakdown）」，那正是 CG 有限步终止的原因。
 * 6. **纵轴 0 在下**：两张矩阵的纵轴都是「索引 i 向上增长」（`Axis` 没有自定义刻度标签，
 *    见 `scene.ts` 的说明）。H 是**上** Hessenberg（i > j+1 处恒 0），在这个朝向下零区
 *    落在左上角、非零阶梯靠右下——文本里点名，免得按「标准矩阵写法」读反。
 * 7. **3D 是投影，不是数据**：`dim = 3` 时把基向量与解轨迹做**固定的**正交投影
 *    （方位角 30°、仰角 20°），投影后的坐标只用来摆放像素；同一批坐标以数字列在中栏——
 *    投影图看不出深度，数字才是可核对的那一份。`dim = 2` 时不投影，直接等比例画。
 */

import { formatNumber, niceLimits, niceTicks } from "../../shared/src/axes.js";
import type { Frame } from "../../shared/src/frame.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  resolveTextOverlaps,
  ModelInputError,
  applyOverflow,
  canvasSize,
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
  type TextSpec,
} from "../../shared/src/scene.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "1.5";
const VIZ = "krylov_growth";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "Krylov 子空间逐步扩张与残差正交性";
/** 矩阵热力图的对数地板：0 与机器零都落在 1e-16（见文件头第 2 条）。 */
const MATRIX_LOG_FLOOR = -16;
/** 舍入底：残差低于 √u·‖r_0‖ 之后归一化内积没有意义（与后端同口径）。 */
const ROUNDING_FLOOR = Math.sqrt(Number.EPSILON);
/** 3D 正交投影的固定视角（显示几何，不是数据）。 */
const AZIMUTH = Math.PI / 6;
const ELEVATION = Math.PI / 9;

/** 固定角标槽位（见 `model.ts::BADGE_LAYOUT`）。本帧数据源恒为 backend，故不占 preview 槽。 */
const SLOT = {
  source: 0,
  axisMode: 1,
  matrixAxis: 2,
  residAxis: 3,
  status: 4,
  overflow: 5,
  degraded: 6,
} as const;

export const krylovGrowthModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组：缺一个就抛 ModelInputError（宁可当场炸，也不画半张图）——
  const basis = frameMatrix(frame, "basis");
  const iterates = frameMatrix(frame, "iterates");
  const orth = frameMatrix(frame, "orthogonality");
  const residual = requireArray(frame, "residual");
  const stepGrid = requireArray(frame, "step_grid");
  const xStar = requireArray(frame, "x_star");
  const dim = xStar.length;
  if (dim < 2 || dim > 3) {
    throw new ModelInputError(
      `帧 ${CHAPTER}/${VIZ} 的 x_star 有 ${dim} 个分量：本图只画 2D/3D（manifest 的 dim 参数是 2...3）`,
    );
  }
  if (basis.cols !== dim) {
    throw new ModelInputError(
      `帧 ${CHAPTER}/${VIZ} 的 basis 是 ${basis.rows}×${basis.cols}，但 x_star 有 ${dim} 维：基向量的列数必须等于空间维数`,
    );
  }
  if (iterates.cols !== dim) {
    throw new ModelInputError(
      `帧 ${CHAPTER}/${VIZ} 的 iterates 是 ${iterates.rows}×${iterates.cols}，但 x_star 有 ${dim} 维：解轨迹的列数必须等于空间维数`,
    );
  }
  if (orth.rows !== orth.cols) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 orthogonality 是 ${orth.rows}×${orth.cols}，不是方阵`);
  }
  if (residual.length > stepGrid.length) {
    throw new ModelInputError(
      `帧 ${CHAPTER}/${VIZ} 的 residual 长度 ${residual.length} 超过 step_grid 的 ${stepGrid.length}：前缀曲线必须落在步网格之内`,
    );
  }
  const pitfalls = requirePitfalls(frame);

  // —— 可选字段：缺了就降级（记进 degraded，最终挂成一个 status 角标）——
  const degraded: string[] = [];
  const tMeta = metaNumber(frame, "t");
  if (tMeta === undefined) degraded.push("帧 meta 缺 t：scene.t 改用请求 t（两者可能不同）");
  const tNow = tMeta ?? opts.t;
  const k = metaNumber(frame, "step");
  if (k === undefined) degraded.push("帧 meta 缺 step：不画当前步标记（不猜是第几步）");
  if (k !== undefined && Number.isFinite(opts.t) && Math.round(opts.t) !== Math.round(k)) {
    degraded.push(
      `请求 t=${formatNumber(opts.t, 4)} 夹到第 ${formatNumber(k, 4)} 步：帧给的是四舍五入并夹到 [0, n_steps] 的整数步`,
    );
  }
  const nSteps = metaNumber(frame, "n_steps");
  const nBasis = metaNumber(frame, "n_basis");
  if (nBasis !== undefined && Math.round(nBasis) !== basis.rows) {
    degraded.push(
      `帧 meta.n_basis=${formatNumber(nBasis, 4)} 与 basis 的 ${basis.rows} 行不一致：按数组行数画（数组是数据）`,
    );
  }
  const subspaceDim = metaNumber(frame, "subspace_dim");
  const kappaMeasured = metaNumber(frame, "kappa_measured");
  const orthMax = metaNumber(frame, "orthogonality_max");
  const basisOrth = metaNumber(frame, "basis_orthogonality");
  const arnoldiOrth = metaNumber(frame, "arnoldi_orthogonality");
  const happy = (metaNumber(frame, "happy_breakdown") ?? 0) > 0.5;
  const nIter = metaNumber(frame, "n_iter");
  const residualFinal = metaNumber(frame, "residual_final");
  const orthName = stringOf(frame, "orthogonality_name");
  if (orthName === undefined) {
    degraded.push("帧 strings 缺 orthogonality_name：正交性矩阵画的是哪一条内积无从得知（不按方法名猜）");
  }
  const methodName = stringOf(frame, "method_name") ?? stringOf(frame, "method");
  const note = stringOf(frame, "note");
  if (note === undefined) degraded.push("帧 strings 缺 note：状态角标不自行编造结论");
  const residualMarkArr = optionalArray(frame, "residual_mark");
  const basisGrid = optionalArray(frame, "basis_grid");
  const hessenberg = frame.has("hessenberg") ? frameMatrix(frame, "hessenberg", true) : undefined;
  if (hessenberg === undefined) degraded.push("帧缺 hessenberg：H 一栏留空（CG 没有 Arnoldi 的 H 是正常的）");
  if (basisOrth === undefined) degraded.push("帧 meta 的 basis_orthogonality 是 null：不显示基的正交性（CG 帧本来就不给）");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tNow,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // —— 排版：左列「空间图 + 残差」；中列 H 与正交性两张矩阵热力图（色标过道在它们右侧）；
  //           右列是唯一的长文字栏（读数与说明）——
  // 三块内容的横向预算：空间图是正方形（等比例的前提）；两张热力图右边要留 118px 给
  // 光栅器画色标条与它的数值标签（Scene 契约里没有色标节点，见 `tools/raster.py`）；
  // 文字栏的左边界必须让开那条色标（heatX + heatW + 46 + 14 + 数值标签宽）。
  const base = stackedPanels(width, height);
  const contentH = base.bottom - base.top;
  const spaceSide = Math.max(150, Math.min(260, contentH - 140));
  const spaceRect: Rect = [base.left, base.top, spaceSide, spaceSide];
  // 残差轴的轴标题带（y−18...y−2）要落在空间图 xlabel 带（+16...+28）之下，否则两块装饰叠在一起。
  const residTop = spaceRect[1] + spaceSide + 64; // 空间图的 xlabel 带(+16…+28)与残差轴的标题带(y−18…y−2)之间留够缝
  const residRect: Rect = [base.left, residTop, spaceSide, Math.max(70, base.bottom - residTop)];
  const heatX = base.left + spaceSide + 40; // 空间图右侧没有装饰带，36px 就够
  const heatW = Math.max(140, Math.min(180, Math.round(base.contentWidth * 0.17)));
  const heatH = Math.max(90, Math.min(120, Math.round(contentH * 0.29)));
  const hessRect: Rect = [heatX, base.top, heatW, heatH];
  const orthRect: Rect = [heatX, base.top + heatH + 62, heatW, heatH];
  // 文字栏左边界 = 色标条 + 它的数值标签（raster 画在 right+46…+104）之后，再留 40px。
  const textX = heatX + heatW + 150;
  const textW = Math.max(140, base.left + base.contentWidth - textX);

  // —— 空间图：2D 直接画，3D 做固定正交投影（见文件头第 7 条）——
  const project = (v: readonly number[]): [number, number] => (dim === 2 ? [v[0] as number, v[1] as number] : project3(v));
  const basisPts: [number, number][] = [];
  for (let j = 0; j < basis.rows; j += 1) basisPts.push(project(rowOf(basis, j)));
  const iterPts: [number, number][] = [];
  for (let i = 0; i < iterates.rows; i += 1) iterPts.push(project(rowOf(iterates, i)));
  const starPt = project(Array.from({ length: dim }, (_, i) => xStar[i] as number));
  const axisStubs: [number, number][][] = [];
  if (dim === 3) {
    let r = 0;
    for (const p of [...basisPts, ...iterPts, starPt]) r = Math.max(r, Math.abs(p[0]), Math.abs(p[1]));
    const lim = Math.max(0.5, r * 1.25);
    for (let i = 0; i < 3; i += 1) {
      const e = [0, 0, 0];
      e[i] = lim;
      axisStubs.push([
        [0, 0],
        project(e),
      ]);
    }
  }
  const spaceBounds = boundsOf([...basisPts, ...iterPts, starPt, [0, 0], ...axisStubs.flat()]);
  const spaceLim = equalSpanLimits(spaceBounds, 4);
  const spaceAxis = makeAxis("space", spaceRect, {
    xlim: spaceLim.xlim,
    ylim: spaceLim.ylim,
    xlabel: dim === 2 ? "x1" : "投影 X（az=30°, el=20°）",
    ylabel: dim === 2 ? axisYlabel("space", "x2") : "投影 Y",
    title: axisTitle("space", "Krylov 基与解轨迹（2D/3D）"),
    autoscaled: true,
    xticks: spaceLim.ticks,
    yticks: spaceLim.ticks,
  });

  // 两张矩阵热力图：**固定轴**（行列都是 0...N−1 的格心，0 在下，见文件头第 6 条）。
  const nOrth = orth.rows;
  const orthLim: [number, number] = [-0.5, Math.max(1.5, nOrth - 0.5)];
  const orthAxis = makeAxis("orth", orthRect, {
    xlim: orthLim,
    ylim: orthLim,
    xlabel: "基 j",
    ylabel: axisYlabel("orth", "步 i（0 在下）/ 基 j"),
    title: axisTitle("orth", "残差正交性实测矩阵"),
    autoscaled: false,
    xticks: niceTicks(0, Math.max(1, nOrth - 1), 5),
    yticks: niceTicks(0, Math.max(1, nOrth - 1), 5),
  });
  const nH = hessenberg === undefined ? 0 : hessenberg.cols;
  const nHRow = hessenberg === undefined ? 0 : hessenberg.rows;
  const hessLim: [number, number] = [-0.5, Math.max(1.5, nHRow - 0.5)];
  const hessXLim: [number, number] = [-0.5, Math.max(1.5, nH - 0.5)];
  const hessAxis = makeAxis("hess", hessRect, {
    xlim: hessXLim,
    ylim: hessLim,
    xlabel: "列 j（0 在左）",
    ylabel: "行 i（0 在下）",
    title: "Arnoldi 的 H（Hessenberg）",
    autoscaled: false,
    xticks: niceTicks(0, Math.max(1, nH - 1), 5),
    yticks: niceTicks(0, Math.max(1, nHRow - 1), 5),
  });

  // 残差轴（本图新增，manifest 只声明了 space/orth）：对数纵轴，范围含 √u 参考线。
  const resPos = finitePositiveMinMax(residual);
  const floorU = ROUNDING_FLOOR; // r_0 在帧里归一化为 1
  // 纵轴范围只看残差自己：把 √u = 1.49e-8 也塞进来会把 1 → 0.44 这一段压成一条水平线，
  // 看图的人会以为「残差根本没动」（GMRES(3) 三步就收敛，本来就是小范围）。
  const hasPositive = resPos !== undefined && Number.isFinite(resPos[0]) && resPos[0] > 0;
  const residYlim: [number, number] = hasPositive
    ? niceLimits(resPos[0], Math.max(resPos[1], resPos[0] * 10), 4, { log: true, pad: 0.25 })
    : [1e-16, 1];
  if (!hasPositive) degraded.push("residual 里没有 >0 的有限值：残差轴用占位范围 [1e-16, 1]，且不画曲线");
  const stepMax = Math.max(1, nSteps ?? residual.length - 1);
  const residAxis = makeAxis("resid", residRect, {
    xlim: [0, stepMax],
    ylim: residYlim,
    yscale: "log",
    xlabel: "Krylov 步 k",
    ylabel: "相对残差",
    title: "相对残差与舍入底（对数）",
    autoscaled: true,
    xticks: niceTicks(0, stepMax, 4),
  });
  scene.axes.push(spaceAxis, hessAxis, orthAxis, residAxis);

  // —— 图层 1~4：空间图 ——
  if (axisStubs.length > 0) {
    scene.layers.push(
      lineLayer(
        "space",
        axisStubs.map(([a, b]) => series([a[0], b[0]], [a[1], b[1]], pal.reference, { width: 0.8, dash: [3, 3] })),
      ),
    );
  }
  if (basisPts.length > 0) {
    scene.layers.push(
      lineLayer(
        "space",
        basisPts.map((p, j) =>
          series([0, p[0]], [0, p[1]], pal.curves[0], {
            width: 1.6,
            label:
              j === basisPts.length - 1
                ? endLabel(p[0], spaceAxis.xlim, spaceRect[2], "Krylov 基 v_j（帧 basis）")
                : undefined,
          }),
        ),
      ),
    );
  }
  if (iterPts.length >= 2) {
    scene.layers.push(
      lineLayer("space", [
        series(
          iterPts.map((p) => p[0]),
          iterPts.map((p) => p[1]),
          pal.exact,
          {
            width: 1.8,
            marker: "o",
            label: endLabel(
              iterPts[iterPts.length - 1]?.[0] ?? Number.NaN,
              spaceAxis.xlim,
              spaceRect[2],
              "解路径 x_k（帧 iterates）",
            ),
          },
        ),
      ]),
    );
  }
  scene.layers.push(scatterLayer("space", [starPt[0]], [starPt[1]], { r: 4.5, color: pal.marker, label: "精确解 x*" }));
  if (iterPts.length > 0) {
    const last = iterPts[iterPts.length - 1] as [number, number];
    scene.layers.push(scatterLayer("space", [last[0]], [last[1]], { r: 3.6, color: pal.exact, label: "当前步 x_k" }));
  }

  // —— 图层 5：Arnoldi 的 H（0×0 的 CG 帧不画；见文件头第 5 条）——
  const hessNote: string[] = [];
  if (hessenberg !== undefined && hessenberg.rows >= 2 && hessenberg.cols >= 2) {
    const hv = toLogFloor(hessenberg.values, MATRIX_LOG_FLOOR);
    scene.layers.push(
      heatLayer("hess", hv, hessenberg.cols, hessenberg.rows, [hessXLim[0], hessXLim[1], hessLim[0], hessLim[1]], {
        cmap: "viridis",
        vmin: MATRIX_LOG_FLOOR,
        vmax: Math.max(MATRIX_LOG_FLOOR + 1, Math.ceil(maxFinite(hv, MATRIX_LOG_FLOOR))),
        logScale: false, // 值已经取过 log10，再开对数就成了 loglog
        label: "log10|H|（地板 1e-16）",
      }),
    );
    hessNote.push(`H 是 ${formatNumber(hessenberg.rows, 3)}×${formatNumber(hessenberg.cols, 3)}，纵轴 0 在下：零区在左上`);
  } else if (hessenberg !== undefined) {
    hessNote.push(`${formatNumber(hessenberg.rows, 3)}×${formatNumber(hessenberg.cols, 3)} 空矩阵：CG 没有 H（heat 至少要 2×2）`);
  }

  // —— 图层 6~7：正交性矩阵 + 舍入底分界线 ——
  const orthFloorLine = noiseFloorRow(residual, floorU);
  const orthValues = toLogFloor(orth.values, MATRIX_LOG_FLOOR);
  if (nOrth >= 2) {
    scene.layers.push(
      heatLayer("orth", orthValues, nOrth, nOrth, [orthLim[0], orthLim[1], orthLim[0], orthLim[1]], {
        cmap: "magma",
        vmin: MATRIX_LOG_FLOOR,
        vmax: Math.max(MATRIX_LOG_FLOOR + 1, Math.ceil(maxFinite(orthValues, MATRIX_LOG_FLOOR))),
        logScale: false,
        label: "log10|内积|",
      }),
    );
  } else {
    degraded.push(`正交性矩阵只有 ${nOrth}×${nOrth}：heat 至少需要 2×2，本帧不画`);
  }
  if (orthFloorLine !== undefined && orthFloorLine > orthLim[0] && orthFloorLine < orthLim[1]) {
    scene.layers.push(
      lineLayer("orth", [
        // 分界线横跨整个面板，末点标签会冲出画布：位置与含义写在面板文本里。
        series([orthLim[0], orthLim[1]], [orthFloorLine, orthFloorLine], pal.warn, { dash: [5, 4], width: 1.2 }),
      ]),
    );
  }

  // —— 图层 8~10：残差曲线 ——
  const cut = k === undefined ? stepGrid.length - 1 : clampInt(k, 0, stepGrid.length - 1);
  const nDraw = Math.min(cut + 1, residual.length, stepGrid.length);
  const residSeries = segmentedSeries(prefixOf(stepGrid, nDraw), prefixOf(residual, nDraw), pal.residual, {
    width: 1.8,
    label: endLabel(prefixOf(stepGrid, nDraw)[nDraw - 1] ?? Number.NaN, residAxis.xlim, residRect[2], "相对残差（帧 residual）"),
    accept: (v) => v > 0,
  });
  const zeroCount = nDraw - countPositive(prefixOf(residual, nDraw));
  const residLayers = [...residSeries];
  // 舍入底参考线落在轴外时明说一句：不画可以，但不能不说（否则读者以为这条线不存在）。
  const floorOutside = !(floorU > residYlim[0] && floorU < residYlim[1]);
  const floorNote =
    hasPositive && floorOutside
      ? `√u = ${formatNumber(floorU, 3)} 在轴外（残差未到舍入底），不画`
      : undefined;
  if (!floorOutside) {
    residLayers.push(
      series([0, stepMax], [floorU, floorU], pal.reference, {
        width: 1,
        dash: [5, 4],
        // 参考线横跨整个面板：标签写在面板文本里（末点标签会溢出画布）。
        label: undefined,
      }),
    );
  }
  if (residLayers.length > 0) scene.layers.push(lineLayer("resid", residLayers));
  const rMark = residualMarkArr !== undefined && residualMarkArr.length > 0 ? (residualMarkArr[0] as number) : metaNumber(frame, "residual");
  if (rMark !== undefined && Number.isFinite(rMark) && rMark > 0) {
    scene.layers.push(scatterLayer("resid", [cut], [rMark], { r: 3.4, color: pal.marker, label: "第 k 步残差（帧 residual_mark）" }));
  }

  // —— 文本 ——
  const spaceCaption: string[] = [];
  spaceCaption.push(
    dim === 3
      ? "左图 = R³ 斜视图：深度不可读，量值看本栏"
      : "左图 = R² 等比例（x/y 同跨度）",
  );
  spaceCaption.push(
    dim === 3
      ? "细虚线 = 坐标轴；蓝 = 基 v_j；黄点线 = 解路径"
      : "蓝 = 基 v_j；黄点线 = 解路径",
  );
  scene.texts.push(
    ...textBlock(
      textX,
      base.top + 2,
      base.bottom - 4,
      panelLines({
        dim,
        methodName,
        k,
        nSteps,
        subspaceDim,
        nBasis,
        basisRows: basis.rows,
        kappaMeasured,
        orthMax,
        basisOrth,
        arnoldiOrth,
        happy,
        nIter,
        residualFinal,
        hessenberg,
        hessNote,
        basisPts,
        iterPts,
        basisGrid,
        starPt,
        note,
        zeroCount,
        caption: spaceCaption,
        floorNote,
      }),
      pal.muted,
      unitBudget(textW),
      "top",
    ),
  );
  const orthLines: string[] = [];
  // 面板标题已经写了矩阵的名字，这里只交代色标与口径，三句压进 5 行。
  orthLines.push("色标 log10|内积|，地板 1e-16。");
  orthLines.push(
    orthMax === undefined
      ? "帧 meta 缺 orthogonality_max。"
      : `下三角最大值 = ${formatNumber(orthMax, 3)}（已屏蔽舍入底以下行）`,
  );
  orthLines.push(
    orthFloorLine === undefined
      ? "纵轴 0 在下；两张矩阵都是整段算完的。"
      : `y>=${formatNumber(orthFloorLine, 3)} 以上：残差低于 √u，是噪声；纵轴 0 在下`,
  );
  scene.texts.push(
    ...textBlock(
      orthRect[0],
      orthRect[1] + orthRect[3] + 30,
      base.bottom - 4,
      orthLines,
      pal.muted,
      unitBudget(orthRect[2] + 118),
      "top",
    ),
  );
  // —— 角标（固定槽位）——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(
    placeBadge(badge("autoscale", `${axisModeBadge(true).text}（space：按本帧基向量与轨迹取整；自适应会掩盖量级变化）`, 8, 8), SLOT.axisMode, width),
  );
  scene.badges.push(
    placeBadge(badge("autoscale", "hess / orth：固定（0...n_iter 的格心，矩阵格子不随数据变）", 8, 8), SLOT.matrixAxis, width),
  );
  scene.badges.push(placeBadge(badge("autoscale", "resid：自适应（log 纵轴按本帧残差与 √u 取整）", 8, 8), SLOT.residAxis, width));
  const status: string[] = [];
  if (methodName !== undefined) status.push(`方法 ${methodName}`);
  if (k !== undefined) status.push(nSteps === undefined ? `第 ${formatNumber(k, 4)} 步` : `第 ${formatNumber(k, 4)}/${formatNumber(nSteps, 4)} 步`);
  if (subspaceDim !== undefined) status.push(`子空间维数 ${formatNumber(subspaceDim, 4)}/${formatNumber(dim, 4)}`);
  if (happy) status.push("! happy breakdown：β 变 0，子空间已不变（提前终止是数学，不是 bug）");
  if (note !== undefined) status.push(note);
  if (status.length === 0) status.push("帧未给出可判读的状态（预期有 meta.step 与 strings.note）");
  scene.badges.push(placeBadge(badge("status", status.join("；"), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(basis.values) +
    countNonFinite(iterates.values) +
    countNonFinite(orth.values) +
    countNonFinite(residual) +
    countNonFinite(stepGrid) +
    countNonFinite(xStar) +
    (hessenberg === undefined ? 0 : countNonFinite(hessenberg.values));
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

/** 二维数组（行主序）。`allowEmpty = true` 时接受 0×0（CG 的 `hessenberg` 就是空矩阵）。 */
function frameMatrix(frame: Frame, name: string, allowEmpty = false): Matrix2D {
  const values = requireArray(frame, name);
  const shape = frame.shapeOf(name);
  if (shape.length !== 2) {
    throw new ModelInputError(
      `帧 ${CHAPTER}/${VIZ} 的 '${name}' 形状 [${shape.join(", ")}] 不是二维：本图按「行 = 一条曲线 / 一行矩阵」消费`,
    );
  }
  const rows = shape[0] as number;
  const cols = shape[1] as number;
  if (values.length !== rows * cols) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 '${name}' 有 ${values.length} 个元素，与 ${rows}×${cols} 不符`);
  }
  if (!allowEmpty && (rows === 0 || cols === 0)) {
    throw new ModelInputError(`帧 ${CHAPTER}/${VIZ} 的 '${name}' 是空矩阵 ${rows}×${cols}：这一帧没有可画的内容`);
  }
  return { rows, cols, values };
}

function rowOf(m: Matrix2D, i: number): number[] {
  const out: number[] = [];
  for (let j = 0; j < m.cols; j += 1) out.push(m.values[i * m.cols + j] as number);
  return out;
}

function prefixOf(a: ArrayLike<number>, n: number): Float64Array {
  const len = Math.max(0, Math.min(n, a.length));
  const out = new Float64Array(len);
  for (let i = 0; i < len; i += 1) out[i] = a[i] as number;
  return out;
}

function countPositive(a: ArrayLike<number>): number {
  let n = 0;
  for (let i = 0; i < a.length; i += 1) if ((a[i] as number) > 0) n += 1;
  return n;
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

/** log10(max(|v|, 10^floor))：0 与极小值都压到地板（见文件头第 2 条）。 */
function toLogFloor(a: ArrayLike<number>, floorLog: number): Float64Array {
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i] as number;
    if (!Number.isFinite(v)) {
      out[i] = Number.NaN; // 非有限值交给渲染层画「无数据」灰，不冒充地板
      continue;
    }
    const lg = Math.log10(Math.abs(v));
    out[i] = lg < floorLog ? floorLog : lg;
  }
  return out;
}

/** 第一行「残差低于底」的上方分界 y 坐标；没有则 undefined。 */
function noiseFloorRow(residual: ArrayLike<number>, floor: number): number | undefined {
  for (let i = 0; i < residual.length; i += 1) {
    const v = residual[i] as number;
    if (Number.isFinite(v) && v <= floor) return i - 0.5;
  }
  return undefined;
}

/** 固定正交投影：先绕 z 转 AZIMUTH，再抬 ELEVATION（只用于摆放像素）。 */
function project3(v: readonly number[]): [number, number] {
  const x = v[0] ?? 0;
  const y = v[1] ?? 0;
  const z = v[2] ?? 0;
  const ca = Math.cos(AZIMUTH);
  const sa = Math.sin(AZIMUTH);
  const ce = Math.cos(ELEVATION);
  const se = Math.sin(ELEVATION);
  return [-sa * x + ca * y, -ca * se * x - sa * se * y + ce * z];
}

/** 等跨度轴范围：xlim 与 ylim 的宽度逐位相等（2D 空间图必须等比例）。 */
function equalSpanLimits(
  b: { x0: number; x1: number; y0: number; y1: number },
  target: number,
): { xlim: [number, number]; ylim: [number, number]; ticks: number[] } {
  const cx = (b.x0 + b.x1) / 2;
  const cy = (b.y0 + b.y1) / 2;
  const span = Math.max(b.x1 - b.x0, b.y1 - b.y0, 1e-6) * 1.15;
  const [x0, x1] = niceLimits(cx - span / 2, cx + span / 2, target);
  const w = x1 - x0;
  return { xlim: [x0, x1], ylim: [cy - w / 2, cy + w / 2], ticks: niceTicks(x0, x1, target) };
}

function boundsOf(pts: readonly [number, number][]): { x0: number; x1: number; y0: number; y1: number } {
  let x0 = Number.POSITIVE_INFINITY;
  let x1 = Number.NEGATIVE_INFINITY;
  let y0 = Number.POSITIVE_INFINITY;
  let y1 = Number.NEGATIVE_INFINITY;
  for (const [x, y] of pts) {
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    if (x < x0) x0 = x;
    if (x > x1) x1 = x;
    if (y < y0) y0 = y;
    if (y > y1) y1 = y;
  }
  if (!Number.isFinite(x0)) return { x0: -1, x1: 1, y0: -1, y1: 1 };
  return { x0, x1, y0, y1 };
}

/**
 * 右侧读数栏：状态数字 + 坐标（投影后）+ β（H 的结构在热力图里）。
 *
 * 这一栏只有 25 行、每行 28 个单位宽：坐标一律 2 位有效数字、按「一行放得下」拼，
 * 长句拆短句。放不下的行会被 `textBlock` 丢掉并点名为「另有 N 行」——那等于白写。
 */
function panelLines(m: {
  dim: number;
  methodName?: string;
  k?: number;
  nSteps?: number;
  subspaceDim?: number;
  nBasis?: number;
  basisRows: number;
  kappaMeasured?: number;
  orthMax?: number;
  basisOrth?: number;
  arnoldiOrth?: number;
  happy: boolean;
  nIter?: number;
  residualFinal?: number;
  hessenberg?: Matrix2D;
  hessNote: string[];
  basisPts: readonly [number, number][];
  iterPts: readonly [number, number][];
  basisGrid?: ArrayLike<number>;
  starPt: [number, number];
  note?: string;
  zeroCount: number;
  caption: string[];
  floorNote?: string;
}): string[] {
  const out: string[] = [];
  const head: string[] = [`n=${formatNumber(m.dim, 3)}`];
  if (m.methodName !== undefined) head.push(`方法 ${m.methodName}`);
  if (m.k !== undefined) {
    head.push(m.nSteps === undefined ? `第 ${formatNumber(m.k, 3)} 步` : `第 ${formatNumber(m.k, 3)}/${formatNumber(m.nSteps, 3)} 步`);
  }
  out.push(head.join("　"));
  const num: string[] = [];
  if (m.subspaceDim !== undefined) num.push(`子空间维数 ${formatNumber(m.subspaceDim, 3)}/${formatNumber(m.dim, 3)}`);
  if (m.kappaMeasured !== undefined) num.push(`κ(A)=${formatNumber(m.kappaMeasured, 4)}`);
  out.push(num.join("　"));
  const mid: string[] = [];
  if (m.nIter !== undefined) mid.push(`整段 ${formatNumber(m.nIter, 3)} 步`);
  if (m.residualFinal !== undefined) mid.push(`末步残差 ${formatNumber(m.residualFinal, 3)}`);
  if (m.happy) mid.push("happy breakdown");
  if (mid.length > 0) out.push(mid.join("　"));
  const ortho: string[] = [];
  if (m.basisOrth !== undefined) ortho.push(`基正交性 ${formatNumber(m.basisOrth, 3)}`);
  if (m.arnoldiOrth !== undefined) ortho.push(`Arnoldi ${formatNumber(m.arnoldiOrth, 3)}`);
  if (m.nBasis !== undefined && m.nBasis !== m.basisRows) ortho.push(`meta.n_basis=${formatNumber(m.nBasis, 3)}（数组 ${m.basisRows} 行）`);
  if (ortho.length > 0) out.push(ortho.join("　"));

  // 坐标列：投影看不出深度，这批数字才是"能对上号"的那一份（2 位有效数字）。
  const fmt = (v: number): string => formatNumber(v, 2);
  const basisTxt = m.basisPts
    .map((p, j) => {
      const idx = m.basisGrid === undefined || j >= m.basisGrid.length ? j : (m.basisGrid[j] as number);
      return `v${formatNumber(idx, 3)}(${fmt(p[0])},${fmt(p[1])})`;
    })
    .join(" ");
  out.push(`投影坐标（2 位有效数字，帧 basis）：${basisTxt === "" ? "（本帧没有基向量）" : basisTxt}`);
  const iterTxt = m.iterPts.map((p, i) => `x${formatNumber(i, 3)}(${fmt(p[0])},${fmt(p[1])})`).join(" ");
  out.push(`解路径（帧 iterates）：${iterTxt === "" ? "（本帧没有迭代点）" : iterTxt}`);
  out.push(`精确解（帧 x_star）：x*(${fmt(m.starPt[0])},${fmt(m.starPt[1])})`);
  if (m.hessenberg !== undefined && m.hessenberg.rows >= 2 && m.hessenberg.cols >= 2) {
    const sub: string[] = [];
    for (let j = 0; j + 1 < m.hessenberg.rows && j < m.hessenberg.cols; j += 1) {
      sub.push(fmt(m.hessenberg.values[(j + 1) * m.hessenberg.cols + j] as number));
    }
    out.push(`次对角元 β：${sub.join("、")}——变 0 = 子空间不变`);
  }
  out.push(...m.hessNote);
  if (m.floorNote !== undefined) out.push(m.floorNote);
  out.push("Krylov 残差没有定常衰减因子（不像 ρ^k），本轴不画理论直线：那是方法性质，不是缺数据");
  if (m.zeroCount > 0) out.push(`${formatNumber(m.zeroCount, 3)} 个样本恰为 0：曲线断开，不是缺数据`);
  out.push(...m.caption);
  if (m.note !== undefined) out.push(`帧 note：${m.note}`);
  return out;
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
