/**
 * `descent_race`（5.1 章 viz 1）：同一张二次型等高线上四条下降路径。
 *
 * 三条防误导纪律
 * ------------
 * 1. **路径按 `meta.k_actual` 切片**：帧里的 path/error 数组是完整历史；scrub 到第 k 步
 *    时只画前 k+1 点。把整条路径一次性画出来会把“动画”变成“答案”，读不出锯齿是在哪一步形成的。
 * 2. **四种方法不是同一个 α**：GD 与动量用各自的步长、BFGS/牛顿自带线搜索。侧栏把
 *    α/α* 与发散线 `step_scale = 1 + 1/κ` 分开写；拿“谁先到”当结论之前必须先看步长口径。
 * 3. **误差副轴是对数**：GD 的线性收敛在 log 轴上是直线，牛顿一步掉到舍入级。
 *    纵轴固定地板 1e-17；`meta.diverged_*` 的方法在图上用红色状态点名。
 *
 * 帧 `time.mode="scrub"`：`scene.t` 取帧 `meta.t`。
 */

import { formatNumber } from "../../shared/src/axes.js";
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
  contourLayer,
  lineLayer,
  makeAxis,
  scatterLayer,
  sourceBadge,
} from "../../shared/src/scene.js";
import {
  TEXT_ROW_H,
  axisTitleOf,
  axisYlabelOf,
  columnsLayout,
  textBlock,
  unitBudget,
} from "../_common/panel.js";
import {
  CHAPTER,
  METHOD_LABEL,
  METHOD_ORDER,
  SLOT,
  cropGrid,
  logLevels,
  methodColors,
  prefix,
  stepIndex,
} from "./common.js";
import { manifest } from "./manifest.gen.js";

const VIZ = "descent_race";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "下降路径赛跑：GD、动量、BFGS 与牛顿";
const ERROR_FLOOR_FALLBACK = 1e-17;

export const descentRaceModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  const cx = requireArray(frame, "contour_x");
  const cy = requireArray(frame, "contour_y");
  const cz = requireArray(frame, "contour_z");
  const iterations = requireArray(frame, "iterations");
  requireSameLength(cx, cy, "帧数组 contour_x 与 contour_y");
  if (cz.length !== cx.length * cy.length) {
    throw new Error(`contour_z 长度 ${cz.length} ≠ ${cx.length}×${cy.length}`);
  }

  const paths: Record<string, { x: ArrayLike<number>; y: ArrayLike<number>; err: ArrayLike<number> }> = {};
  for (const name of METHOD_ORDER) {
    const px = requireArray(frame, `path_x_${name}`);
    const py = requireArray(frame, `path_y_${name}`);
    const er = requireArray(frame, `error_${name}`);
    requireSameLength(px, py, `path_x_${name} 与 path_y_${name}`);
    requireSameLength(iterations, er, `iterations 与 error_${name}`);
    paths[name] = { x: px, y: py, err: er };
  }
  const pitfalls = requirePitfalls(frame);

  const degraded: string[] = [];
  const t = metaNumber(frame, "t") ?? opts.t;
  const kMeta = metaNumber(frame, "k_actual");
  if (kMeta === undefined) degraded.push("帧 meta 缺 k_actual：按请求时刻取整步，可能多画半步");
  const kMax = Math.max(0, iterations.length - 1);
  const k = Math.max(0, Math.min(kMax, kMeta ?? stepIndex(t, kMax)));

  const floorMeta = metaNumber(frame, "error_floor");
  const errFloor = floorMeta !== undefined && floorMeta > 0 ? floorMeta : ERROR_FLOOR_FALLBACK;
  if (floorMeta === undefined) degraded.push("帧 meta 缺 error_floor：误差地板用登记常量 1e-17");

  const kappa = metaNumber(frame, "kappa_measured");
  const alpha = metaNumber(frame, "alpha_used");
  const alphaStar = metaNumber(frame, "alpha_star");
  const alphaOver = metaNumber(frame, "alpha_over_stable");
  const stepScale = paramNumber(opts.params, "step_scale");
  const divergeLine = kappa !== undefined && kappa > 0 ? 1 + 1 / kappa : undefined;

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = columnsLayout(scene, width, height, { topShare: 0.6, auxH: 150, gap: 64 });

  // 显示窗包住四条路径（起点 (1,1)、动量过冲到 y≈−0.23），不再用 ±2.2 的空白
  // 把轨迹压成一团。横纵轴像素尺度不同，灰线只作 f 的底图，不当几何测量。
  const main = makeAxis("main", [L.left, L.topY, L.leftW, L.topH], {
    xlim: [-0.4, 1.7],
    ylim: [-0.5, 1.6],
    xlabel: "x_1",
    ylabel: axisYlabelOf(SPEC?.axes, "main", "x_2"),
    title: axisTitleOf(SPEC?.axes, "main", "等高线 f(x)=const 与四条下降路径"),
    autoscaled: false,
  });

  // 误差副轴：x 取 iterations 的前 k+1 点，y 只由当前可见的历史决定。
  let errHi = 0;
  for (const name of METHOD_ORDER) {
    const er = prefix(paths[name]!.err, k + 1);
    for (let i = 0; i < er.length; i += 1) {
      const v = er[i] as number;
      if (Number.isFinite(v) && v > errHi) errHi = v;
    }
  }
  if (!(errHi > 0)) errHi = 1;
  const aux = makeAxis("aux", [L.left, L.botY, L.leftW, L.botH], {
    xlim: [0, Math.max(1, kMax)],
    ylim: [errFloor, Math.max(2, errHi * 1.6)],
    yscale: "log",
    xlabel: "迭代 k",
    ylabel: axisYlabelOf(SPEC?.axes, "aux", "‖x_k - x*‖"),
    title: axisTitleOf(SPEC?.axes, "aux", "到最优解的距离（对数轴）"),
    autoscaled: true,
  });
  scene.axes.push(main, aux);

  // 主图：等高线只作底图（灰色细线），四色路径与当前点才是数据。
  const levels = logLevels(cz, 13);
  const crop = cropGrid(cx, cy, cz, main.xlim, main.ylim);
  if (crop !== undefined) {
    scene.layers.push(
      contourLayer("main", crop.values, crop.nx, crop.ny, [crop.x0, crop.x1, crop.y0, crop.y1], {
        levels,
        color: pal.reference,
        width: 0.8,
        label: "f(x)=const",
      }),
    );
  } else {
    degraded.push("等高线网格与视窗没有交集：底图省略（轨迹仍画）");
  }
  scene.layers.push(
    lineLayer("aux", {
      x: [aux.xlim[0], aux.xlim[1]],
      y: [errFloor, errFloor],
      color: pal.reference,
      width: 1,
      dash: [3, 3],
    }),
  );

  const colors = methodColors(pal);
  const pathSummary: string[] = [];
  let nFinitePoints = 0;
  for (const name of METHOD_ORDER) {
    const tr = paths[name]!;
    const m = Math.max(1, Math.min(tr.x.length, k + 1));
    const px = prefix(tr.x, m);
    const py = prefix(tr.y, m);
    const er = prefix(tr.err, k + 1);
    let last = -1;
    for (let i = px.length - 1; i >= 0; i -= 1) {
      if (Number.isFinite(px[i] as number) && Number.isFinite(py[i] as number)) {
        last = i;
        break;
      }
    }
    if (px.length >= 2) {
      scene.layers.push(lineLayer("main", { x: px, y: py, color: colors[name], width: name === "gd" ? 1.9 : 1.5, label: METHOD_LABEL[name] }));
    }
    if (last >= 0) {
      scene.layers.push(
        scatterLayer("main", [px[last] as number], [py[last] as number], {
          r: name === "newton" ? 3.4 : 3,
          color: colors[name],
          label: `${METHOD_LABEL[name]} 当前点`,
        }),
      );
      nFinitePoints += 1;
    }
    if (er.length >= 2) {
      scene.layers.push(lineLayer("aux", { x: prefix(iterations, er.length), y: er, color: colors[name], width: name === "gd" ? 1.9 : 1.5, label: METHOD_LABEL[name] }));
    }
    const eNow = er.length > 0 ? (er[er.length - 1] as number) : Number.NaN;
    pathSummary.push(
      `${METHOD_LABEL[name]}：k=${m - 1}，e=${formatNumber(eNow, 3)}，停=${stringOf(frame, `stop_${name}`) ?? "?"}`,
    );
  }

  // 侧栏：参数摘要 + 方法核对表 + 口径。
  const param = paramSummary(frame, opts, [
    { name: "kappa_measured", label: "κ", digits: 4 },
    { name: "alpha_over_stable", label: "α/α*", digits: 4 },
  ]);
  degraded.push(...param.degraded);
  const notes: string[] = [
    `第 k=${k} 步 / 共 ${kMax} 步（时间轴 tMax=60）`,
    param.text,
    kappa !== undefined ? `发散线 step_scale = 1+1/κ = ${formatNumber(divergeLine ?? Number.NaN, 4)}` : "发散线：帧缺 κ，无法给出",
    ...pathSummary,
  ];
  if (alpha !== undefined && alphaStar !== undefined) {
    notes.push(`GD 实际步长 α=${formatNumber(alpha, 3)}，最优 α*=${formatNumber(alphaStar, 3)}`);
  }
  const budget = unitBudget(L.rightW);
  if (L.rightW >= 200) {
    let y = L.topY;
    const push = (lines: readonly string[], color: string): void => {
      const block = textBlock(L.rightX, y, L.floor, lines, color, budget, "top");
      scene.texts.push(...block);
      y += block.length * TEXT_ROW_H;
    };
    push([notes[0] as string, notes[1] as string], pal.ink);
    if (divergeLine !== undefined) {
      push(
        [notes[2] as string],
        alphaOver !== undefined && alphaOver > 1 ? pal.warn : pal.muted,
      );
    }
    // 每条方法一块，用对应颜色；块高按真实折行数推进，避免窄列里互相压住。
    for (let i = 0; i < pathSummary.length; i += 1) {
      push([pathSummary[i] as string], colors[METHOD_ORDER[i]!]);
    }
    if (y + 4 * TEXT_ROW_H < L.floor) {
      push(
        [
          "步长口径：GD α，动量 (β*,α*)。",
          "BFGS/牛顿用线搜索，不与前两者比步长。",
          alphaOver !== undefined && alphaOver > 1
            ? "α/α*>1：GD 仍可能收敛（发散线见上）。"
            : "α/α*≤1：GD 在稳定区；锯齿由 κ 定。",
          "路径只画到当前步；灰线只是底图。",
          "横纵像素尺度不同，别拿它量角度。",
        ],
        pal.muted,
      );
    }
  } else {
    degraded.push("右侧说明列过窄：方法核对表并入角标，详细口径见 pitfalls");
  }

  // 用“当前误差”拼一条状态角标；文本短，避免被裁剪成看不懂的残句。
  let eGd = Number.NaN;
  const eg = prefix(paths.gd!.err, k + 1);
  if (eg.length > 0) eGd = eg[eg.length - 1] as number;
  scene.badges.push(placeBadge(sourceBadge(dataSourceOf(frame)), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge(
        "status",
        `k=${k}/${kMax}｜κ=${formatNumber(kappa ?? Number.NaN, 4)}｜α/α*=${formatNumber(alphaOver ?? Number.NaN, 3)}｜GD e=${formatNumber(eGd, 3)}`,
        8,
        8,
      ),
      SLOT.status,
      width,
    ),
  );
  if (stepScale !== undefined && divergeLine !== undefined) {
    scene.badges.push(placeBadge(criticalBadge("step_scale", stepScale, divergeLine, { beyond: "above" }), SLOT.critical, width));
  }
  const degradedBadgeItem = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedBadgeItem !== undefined) scene.badges.push(degradedBadgeItem);

  applyOverflow(scene, 0, { slot: SLOT.overflow, width });
  if (nFinitePoints === 0) {
    throw new Error("四条路径在当前步都没有有限点：无法画出任何当前标记");
  }
  requireDrawableLayers(scene);
  return scene;
};
