/**
 * `ghia_benchmark`（6.1 章 viz 2）：Ghia 中心线剖面 + 网格收敛。
 *
 * 三条纪律
 * -------
 * 1. **基准有自己的抄写误差**：Ghia 表跨转载的分歧在 1e-2 量级（`transcription_floor`），
 *    页面上把这条地板画出来；RMSE 低于它时，再压容差量的是转载版本。
 * 2. **插值方向固定**：数值解插到 Ghia 原表节点（`u_num_ref`），不把基准插到细网格上——
 *    后者会把离散误差抹平。
 * 3. **两条剖面各有主次**：左侧 u(y) 是主对拍；右侧 v(x) 是第二个视角。误差曲线同时给
 *    总 RMSE、u、v 三条，别只看总数。
 *
 * 帧 `time.mode="none"`。
 */

import {
  applyOverflow,
  canvasSize,
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
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  lineLayer,
  makeAxis,
  scatterLayer,
  sourceBadge,
} from "../../shared/src/scene.js";
import { axisTitleOf, axisYlabelOf, columnsLayout } from "../_common/panel.js";
import { CHAPTER, SLOT, extentOf, fmt, padRange, pushGroups, singleColumn, statusLine } from "./common.js";
import { manifest } from "./manifest.gen.js";

const VIZ = "ghia_benchmark";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "Ghia 基准：中心线剖面与网格收敛";

export const ghiaBenchmarkModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  const yRef = requireArray(frame, "y_ref");
  const uRef = requireArray(frame, "u_ref");
  const uNum = requireArray(frame, "u_num_ref");
  const xRef = requireArray(frame, "x_ref");
  const vRef = requireArray(frame, "v_ref");
  const vNum = requireArray(frame, "v_num_ref");
  const yProfile = requireArray(frame, "y_profile");
  const uProfile = requireArray(frame, "u_profile");
  const xProfile = requireArray(frame, "x_profile");
  const vProfile = requireArray(frame, "v_profile");
  const nGrid = requireArray(frame, "n_grid");
  const err = requireArray(frame, "error");
  const errU = requireArray(frame, "error_u");
  const errV = requireArray(frame, "error_v");
  requireSameLength(yRef, uRef, "y_ref 与 u_ref");
  requireSameLength(yRef, uNum, "y_ref 与 u_num_ref");
  requireSameLength(xRef, vRef, "x_ref 与 v_ref");
  requireSameLength(xRef, vNum, "x_ref 与 v_num_ref");
  requireSameLength(yProfile, uProfile, "y_profile 与 u_profile");
  requireSameLength(xProfile, vProfile, "x_profile 与 v_profile");
  requireSameLength(nGrid, err, "n_grid 与 error");
  requireSameLength(nGrid, errU, "n_grid 与 error_u");
  requireSameLength(nGrid, errV, "n_grid 与 error_v");
  const pitfalls = requirePitfalls(frame);

  const degraded: string[] = [];
  const re = metaNumber(frame, "re");
  const n = metaNumber(frame, "n");
  const rmse = metaNumber(frame, "rmse");
  const rmseU = metaNumber(frame, "rmse_u");
  const rmseV = metaNumber(frame, "rmse_v");
  const floor = metaNumber(frame, "transcription_floor");
  const slope = metaNumber(frame, "grid_slope");
  const slopeU = metaNumber(frame, "grid_slope_u");
  const slopeV = metaNumber(frame, "grid_slope_v");
  const bestN = metaNumber(frame, "best_n");
  const centerDist = metaNumber(frame, "center_dist");
  const residual = metaNumber(frame, "residual");
  const converged = (metaNumber(frame, "converged") ?? 0) > 0.5;
  const status = (frame.strings["status"] as string | undefined) ?? "?";
  if (slope === undefined) degraded.push("帧缺 grid_slope：侧栏不报收敛阶");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: 0,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const wide = width >= 820;
  const L = wide ? columnsLayout(scene, width, height, { topShare: 0.6, auxH: 150, gap: 56 }) : null;
  const S = wide ? null : singleColumn(scene, width, height, 0.58);
  const topY = L ? L.topY : (S as NonNullable<typeof S>).topY;
  const floorY = L ? L.floor : (S as NonNullable<typeof S>).floor;
  const mainRect: [number, number, number, number] = L ? [L.left, L.topY, L.leftW, L.topH] : (S as NonNullable<typeof S>).main;
  const errRect: [number, number, number, number] = L ? [L.left, L.botY, L.leftW, L.botH] : (S as NonNullable<typeof S>).aux;

  const uRange = extentOf([uRef, uNum, uProfile], [-0.2, 1]);
  const main = makeAxis("main", mainRect, {
    xlim: padRange(Math.min(0, uRange[0]), Math.max(0, uRange[1]), 0.04),
    ylim: [0, 1],
    xlabel: "$u$",
    ylabel: axisYlabelOf(SPEC?.axes, "main", "y"),
    title: axisTitleOf(SPEC?.axes, "main", "竖直中心线 u(y) vs Ghia"),
    autoscaled: false,
    tickTarget: 4,
  });
  const errPos: number[] = [];
  for (let i = 0; i < nGrid.length; i += 1) {
    errPos.push(err[i] as number, errU[i] as number, errV[i] as number);
  }
  const errRange = extentOf([errPos], [1e-4, 1e-2]);
  const yLo = Math.max(1e-12, Math.min(errRange[0], floor ?? errRange[0]) * 0.6);
  const yHi = Math.max(yLo * 20, errRange[1] * 2.2);
  const errAxis = makeAxis("error", errRect, {
    xlim: [Math.max(1, (nGrid[0] as number) - 2), (nGrid[nGrid.length - 1] as number) + 2],
    ylim: [yLo, yHi],
    yscale: "log",
    xlabel: "网格 $n$",
    ylabel: axisYlabelOf(SPEC?.axes, "error", "RMSE"),
    title: axisTitleOf(SPEC?.axes, "error", "RMSE 随网格 n 的收敛（对数纵轴）"),
    autoscaled: true,
    tickTarget: 3,
  });
  scene.axes.push(main, errAxis);

  scene.layers.push(
    lineLayer("main", { x: uNum, y: yRef, color: pal.curves[0] as string, width: 1.9, label: "数值解 $u(y)$" }),
    scatterLayer("main", uRef, yRef, { r: 3.3, color: pal.exact, label: "Ghia 基准" }),
  );
  // 细网格剖面只作背景：它说明线不是“只连了 17 个点”，但不参与对拍。
  scene.layers.push(lineLayer("main", { x: uProfile, y: yProfile, color: pal.curves[0] as string, width: 0.7, alpha: 0.28, label: "迭代轨迹" }));

  let profileAxis: ReturnType<typeof makeAxis> | undefined;
  if (L) {
    const profileH = Math.max(100, Math.min(145, Math.round((floorY - topY) * 0.36)));
    const vRange = extentOf([vRef, vNum, vProfile], [-0.3, 0.2]);
    profileAxis = makeAxis("vprofile", [L.rightX, topY, L.rightW, profileH], {
      xlim: [0, 1],
      ylim: padRange(vRange[0], vRange[1], 0.08),
      xlabel: "$x$",
      ylabel: axisYlabelOf(SPEC?.axes, "vprofile", "v"),
      title: axisTitleOf(SPEC?.axes, "vprofile", "水平中心线 v(x) vs Ghia"),
      autoscaled: false,
      tickTarget: 4,
    });
    scene.axes.push(profileAxis);
    scene.layers.push(
      lineLayer("vprofile", { x: xRef, y: vNum, color: pal.curves[0] as string, width: 1.8, label: "数值解 $v(x)$" }),
      scatterLayer("vprofile", xRef, vRef, { r: 3, color: pal.exact, label: "Ghia 基准 $v$" }),
      lineLayer("vprofile", {
        x: [0, 1],
        y: [0, 0],
        color: pal.reference,
        width: 0.8,
        dash: [3, 3],
      }),
    );
  } else {
    degraded.push("窄画布：省略 v(x) 剖面与右栏说明");
  }

  scene.layers.push(
    lineLayer("error", { x: nGrid, y: err, color: pal.curves[0] as string, width: 2, label: "总 RMSE" }),
    lineLayer("error", { x: nGrid, y: errU, color: pal.exact, width: 1.2, dash: [4, 3], label: "$u$ 剖面的 RMSE" }),
    lineLayer("error", { x: nGrid, y: errV, color: pal.curves[1] as string, width: 1.2, dash: [2, 3], label: "$v$ 剖面的 RMSE" }),
  );
  for (const [arr, color] of [
    [err, pal.curves[0] as string],
    [errU, pal.exact],
    [errV, pal.curves[1] as string],
  ] as const) {
    for (let i = 0; i < nGrid.length; i += 1) {
      const v = arr[i] as number;
      if (!Number.isFinite(v) || v <= 0) continue;
      scene.layers.push(scatterLayer("error", [nGrid[i] as number], [v], { r: 2.2, color }));
    }
  }
  if (floor !== undefined && floor > 0 && floor >= yLo && floor <= yHi) {
    scene.layers.push(
      lineLayer("error", { x: [errAxis.xlim[0], errAxis.xlim[1]], y: [floor, floor], color: pal.reference, width: 1.1, dash: [5, 4], label: "RMSE 地板" }),
    );
  }

  if (L && L.rightW >= 200 && profileAxis !== undefined) {
    const textTop = profileAxis.rect[1] + profileAxis.rect[3] + 36;
    pushGroups(
      scene,
      L.rightX,
      textTop,
      floorY,
      L.rightW,
      [
        {
          lines: [
            `Re=${fmt(re, 0)}，n=${fmt(n, 0)}；RMSE=${fmt(rmse, 4)}。`,
            `u=${fmt(rmseU, 4)}；v=${fmt(rmseV, 4)}。`,
            `残差 ${fmt(residual, 1)}；${converged ? "已收敛" : "未收敛"}。`,
          ],
          color: pal.ink,
        },
        {
          lines: [
            `网格阶 ${fmt(slope, 2)}（log RMSE vs log h）。`,
            `u ${fmt(slopeU, 2)}；v ${fmt(slopeV, 2)}；最优 n=${fmt(bestN, 0)}。`,
            `主涡距 Ghia ${fmt(centerDist, 3)}（O(h)）。`,
          ],
          color: pal.muted,
        },
        {
          lines: [
            "蓝=数值；橙点=Ghia。",
            `灰虚=抄写地板 ${fmt(floor, 1)}。`,
            "误差图：蓝=总，橙虚=u，绿虚=v。",
            "数值解插到 Ghia 原表节点。",
            `状态：${status}。`,
          ],
          color: pal.muted,
        },
      ],
      pal,
    );
  }

  scene.badges.push(placeBadge(sourceBadge(dataSourceOf(frame)), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge(
        "status",
        statusLine([
          `Re=${fmt(re, 0)}`,
          `n=${fmt(n, 0)}`,
          `RMSE ${fmt(rmse, 3)}`,
          `阶 ${fmt(slope, 2)}`,
        ]),
        8,
        8,
      ),
      SLOT.status,
      width,
    ),
  );
  const dbg = degradedBadge(degraded, SLOT.degraded, width);
  if (dbg !== undefined) scene.badges.push(dbg);

  applyOverflow(scene, 0, { slot: SLOT.overflow, width });
  requireDrawableLayers(scene);
  return scene;
};
