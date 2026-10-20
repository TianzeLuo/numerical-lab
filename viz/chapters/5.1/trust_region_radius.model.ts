/**
 * `trust_region_radius`（5.1 章 viz 4）：信赖域半径的自适应。
 *
 * 三条读图纪律
 * ----------
 * 1. **半径只有两种结局**：接受且撞到边界（ρ>t2）才扩张；ρ<t1 收缩；其余不动。
 *    副轴把 Δ_k 与 ρ_k 分开放：ρ 可以为负（这一步会让 f 上升），对数轴画不出负值，
 *    所以 ρ 走线性轴，Δ_k/‖∇f‖ 走对数轴。
 * 2. **被拒绝的步不进轨迹**，但 ratio/accepted 是逐步的历史；图上把接受（绿点）与
 *    拒绝（红点）都放在 ρ 轴上，轨迹点数少于步数是正常的，不是丢数据。
 * 3. **信赖域圆是显示辅助，不是数据**：圆可能被固定窗口裁掉一截；半径的读数在副轴
 *    与状态角标里，圆的视觉大小只用于与当前点对照。
 *
 * 帧 `time.mode="scrub"`：`scene.t` 取帧 `meta.t`。
 */

import { formatNumber } from "../../shared/src/axes.js";
import {
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
  stringOf,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
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
import { CHAPTER, SLOT, circleSeries, logLevels, prefix, stepIndex } from "./common.js";
import { manifest } from "./manifest.gen.js";

const VIZ = "trust_region_radius";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "信赖域半径：接受、拒绝与边界";

export const trustRegionRadiusModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  const cx = requireArray(frame, "contour_x");
  const cy = requireArray(frame, "contour_y");
  const cz = requireArray(frame, "contour_z");
  const trajX = requireArray(frame, "traj_x");
  const trajY = requireArray(frame, "traj_y");
  const iterations = requireArray(frame, "iterations");
  const gnorm = requireArray(frame, "gradient_residual");
  const radius = requireArray(frame, "radius_history");
  const ratio = requireArray(frame, "ratio_history");
  const ratioValid = requireArray(frame, "ratio_valid");
  const accepted = requireArray(frame, "accepted");
  const negCurv = requireArray(frame, "neg_curv");
  requireSameLength(trajX, trajY, "traj_x 与 traj_y");
  requireSameLength(iterations, gnorm, "iterations 与 gradient_residual");
  requireSameLength(iterations, radius, "iterations 与 radius_history");
  requireSameLength(ratio, ratioValid, "ratio_history 与 ratio_valid");
  requireSameLength(ratio, accepted, "ratio_history 与 accepted");
  requireSameLength(ratio, negCurv, "ratio_history 与 neg_curv");
  if (cz.length !== cx.length * cy.length) {
    throw new Error(`contour_z 长度 ${cz.length} ≠ ${cx.length}×${cy.length}`);
  }
  const pitfalls = requirePitfalls(frame);

  const degraded: string[] = [];
  const t = metaNumber(frame, "t") ?? opts.t;
  const kMeta = metaNumber(frame, "k_actual");
  if (kMeta === undefined) degraded.push("帧 meta 缺 k_actual：按请求时刻取整步");
  const kMax = Math.max(0, iterations.length - 1);
  const k = Math.max(0, Math.min(kMax, kMeta ?? stepIndex(t, kMax)));

  const tx = prefix(trajX, k + 1);
  const ty = prefix(trajY, k + 1);
  const rHist = prefix(radius, k + 1);
  const gHist = prefix(gnorm, k + 1);
  const rhoHist = prefix(ratio, Math.min(k, ratio.length));
  const rhoOk = prefix(ratioValid, rhoHist.length);
  const accHist = prefix(accepted, rhoHist.length);
  const ncHist = prefix(negCurv, rhoHist.length);

  const radiusFloorMeta = metaNumber(frame, "radius_floor");
  const errFloorMeta = metaNumber(frame, "error_floor");
  const radiusFloor = radiusFloorMeta !== undefined && radiusFloorMeta > 0 ? radiusFloorMeta : 1e-14;
  const errFloor = errFloorMeta !== undefined && errFloorMeta > 0 ? errFloorMeta : 1e-17;
  if (radiusFloorMeta === undefined) degraded.push("帧 meta 缺 radius_floor：半径地板用 1e-14");
  if (errFloorMeta === undefined) degraded.push("帧 meta 缺 error_floor：梯度地板用 1e-17");

  const delta0 = metaNumber(frame, "delta0");
  const t2 = metaNumber(frame, "t2");
  const radiusNowMeta = metaNumber(frame, "radius_current");
  const radiusNow = radiusNowMeta ?? (rHist.length > 0 ? (rHist[rHist.length - 1] as number) : delta0);
  const ratioNow = metaNumber(frame, "ratio_current");
  const acceptedNow = frame.meta["accepted_current"] === true;
  const nAccepted = metaNumber(frame, "n_accepted");
  const nRejected = metaNumber(frame, "n_rejected");
  const nNegCurv = metaNumber(frame, "n_negative_curv");
  const finalDist = metaNumber(frame, "final_dist");
  const overflow = frame.meta["overflow"] === true;

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = columnsLayout(scene, width, height, { topShare: 0.6, auxH: 150, gap: 64 });
  const main = makeAxis("main", [L.left, L.topY, L.leftW, L.topH], {
    xlim: [-3.5, 3.5],
    ylim: [-1.5, 12],
    xlabel: "x",
    ylabel: axisYlabelOf(SPEC?.axes, "main", "y"),
    title: axisTitleOf(SPEC?.axes, "main", "Rosenbrock 等高线与信赖域圆"),
    autoscaled: false,
  });

  // 窄画布上左列放不下并排的 Δ/‖∇f‖ 与 ρ 两根轴：把 aux 占满左列，ρ 轴明确省略。
  const narrow = L.leftW < 320;
  if (narrow) degraded.push("窄画布：ρ 的独立线性轴省略（接受/拒绝读数在正文与角标里）");
  const auxW = narrow ? L.leftW : Math.max(150, Math.round(L.leftW * 0.56));
  const rateX = L.left + auxW + 54;
  const rateW = narrow ? 0 : Math.max(110, L.leftW - auxW - 54);
  let auxLo = Number.POSITIVE_INFINITY;
  let auxHi = 0;
  for (let i = 0; i < rHist.length; i += 1) {
    for (const v of [rHist[i] as number, i < gHist.length ? (gHist[i] as number) : Number.NaN]) {
      if (Number.isFinite(v) && v > 0) {
        auxLo = Math.min(auxLo, v);
        auxHi = Math.max(auxHi, v);
      }
    }
  }
  const auxYLo = Math.max(Math.min(radiusFloor, errFloor), (Number.isFinite(auxLo) ? auxLo : 1) * 0.3);
  const auxYHi = Math.max(auxYLo * 10, (auxHi > 0 ? auxHi : 1) * 2);
  const aux = makeAxis("aux", [L.left, L.botY, auxW, L.botH], {
    xlim: [0, kMax],
    ylim: [auxYLo, auxYHi],
    yscale: "log",
    xlabel: "",
    ylabel: "Δ_k / ‖∇f‖",
    title: "半径与梯度（对数）",
    autoscaled: true,
  });
  let rate: ReturnType<typeof makeAxis> | undefined;
  if (!narrow) {
    // ρ_k = 实际下降 / 模型下降。t2 最大 1.5 时接受步可以明显大于 1，
    // 原来钉死 [-5,2] 会把几个接受点裁到图外（审计实际抓到过）。
    // 固定窗口仍保留，但上下界按本帧的有效 ρ 扩到至少 [-5,2]。
    let rhoLo = -5;
    let rhoHi = 2;
    for (let i = 0; i < rhoHist.length; i += 1) {
      if ((rhoOk[i] as number) <= 0.5 || !Number.isFinite(rhoHist[i] as number)) continue;
      const v = rhoHist[i] as number;
      rhoLo = Math.min(rhoLo, v - 0.2);
      rhoHi = Math.max(rhoHi, v + 0.2);
    }
    if (t2 !== undefined) rhoHi = Math.max(rhoHi, t2 + 0.2);
    rate = makeAxis("ratio", [rateX, L.botY, rateW, L.botH], {
      xlim: [0, kMax],
      ylim: [rhoLo, rhoHi],
      xlabel: "迭代 k",
      ylabel: "ρ_k",
      title: "ρ_k（线性；窗口随本帧 ρ 扩到 [-5,2] 之外）",
      autoscaled: false,
    });
  }
  scene.axes.push(main, aux);
  if (rate !== undefined) scene.axes.push(rate);

  scene.layers.push(
    contourLayer("main", cz, cx.length, cy.length, [cx[0] as number, cx[cx.length - 1] as number, cy[0] as number, cy[cy.length - 1] as number], {
      levels: logLevels(cz, 11),
      color: pal.reference,
      width: 0.8,
      label: "f(x,y)=const",
    }),
  );
  if (tx.length >= 2) {
    scene.layers.push(lineLayer("main", { x: tx, y: ty, color: pal.marker, width: 1.9, label: "信赖域轨迹" }));
  }
  if (tx.length > 0) {
    const last = tx.length - 1;
    const px = tx[last] as number;
    const py = ty[last] as number;
    if (Number.isFinite(px) && Number.isFinite(py) && Number.isFinite(radiusNow ?? Number.NaN)) {
      const ring = circleSeries(px, py, radiusNow as number, pal.exact, 73);
      ring.width = 1.6;
      ring.dash = [5, 3];
      scene.layers.push(lineLayer("main", [ring]));
    }
    if (Number.isFinite(px) && Number.isFinite(py)) {
      scene.layers.push(
        scatterLayer("main", [px], [py], { r: 4, color: pal.marker, label: "当前点" }),
        scatterLayer("main", [tx[0] as number], [ty[0] as number], { r: 3, color: pal.muted, label: "起点" }),
      );
    }
  }

  scene.layers.push(
    lineLayer("aux", { x: prefix(iterations, rHist.length), y: rHist, color: pal.residual, width: 1.8, label: "信赖域半径 Δ_k" }),
    lineLayer("aux", { x: prefix(iterations, gHist.length), y: gHist, color: pal.curves[0] as string, width: 1.6, label: "梯度范数 ‖∇f‖" }),
  );
  if (radiusFloor >= (aux.ylim[0] as number) && radiusFloor <= (aux.ylim[1] as number)) {
    scene.layers.push(
      lineLayer("aux", { x: [aux.xlim[0], aux.xlim[1]], y: [radiusFloor, radiusFloor], color: pal.reference, width: 1, dash: [3, 3], label: "半径地板" }),
    );
  }
  if (errFloor >= (aux.ylim[0] as number) && errFloor <= (aux.ylim[1] as number)) {
    scene.layers.push(
      lineLayer("aux", { x: [aux.xlim[0], aux.xlim[1]], y: [errFloor, errFloor], color: pal.warn, width: 1, dash: [2, 4], label: "梯度地板" }),
    );
  }
  if (rHist.length > 0) {
    const last = rHist.length - 1;
    const xr = prefix(iterations, rHist.length)[last] as number;
    const lastG = gHist.length - 1;
    const xg = lastG >= 0 ? (prefix(iterations, gHist.length)[lastG] as number) : Number.NaN;
    if (Number.isFinite(xr) && Number.isFinite(rHist[last] as number)) {
      scene.layers.push(scatterLayer("aux", [xr], [rHist[last] as number], { r: 3, color: pal.residual, label: "当前 Δ_k" }));
    }
    if (lastG >= 0 && Number.isFinite(xg) && Number.isFinite(gHist[lastG] as number)) {
      scene.layers.push(scatterLayer("aux", [xg], [gHist[lastG] as number], { r: 3, color: pal.curves[0] as string, label: "当前 ‖∇f‖" }));
    }
  }

  // ρ 轴：线只连“比值有定义”的步；接受/拒绝用点区分。
  const rhoX = prefix(iterations, rhoHist.length);
  if (rate !== undefined && rhoHist.length >= 2) {
    const sx: number[] = [];
    const sy: number[] = [];
    for (let i = 0; i < rhoHist.length; i += 1) {
      if ((rhoOk[i] as number) > 0.5 && Number.isFinite(rhoHist[i] as number)) {
        sx.push(rhoX[i] as number);
        sy.push(rhoHist[i] as number);
      }
    }
    if (sx.length >= 2) scene.layers.push(lineLayer("ratio", { x: sx, y: sy, color: pal.muted, width: 1.2 }));
  }
  const accX: number[] = [];
  const accY: number[] = [];
  const rejX: number[] = [];
  const rejY: number[] = [];
  const ncX: number[] = [];
  const ncY: number[] = [];
  for (let i = 0; rate !== undefined && i < rhoHist.length; i += 1) {
    if ((rhoOk[i] as number) <= 0.5 || !Number.isFinite(rhoHist[i] as number)) continue;
    const x = rhoX[i] as number;
    const y = rhoHist[i] as number;
    if ((accHist[i] as number) > 0.5) {
      accX.push(x);
      accY.push(y);
    } else {
      rejX.push(x);
      rejY.push(y);
    }
    if ((ncHist[i] as number) > 0.5) {
      ncX.push(x);
      ncY.push(y);
    }
  }
  if (rate !== undefined) {
    if (accX.length > 0) scene.layers.push(scatterLayer("ratio", accX, accY, { r: 2.8, color: pal.curves[1] as string, label: "接受" }));
    if (rejX.length > 0) scene.layers.push(scatterLayer("ratio", rejX, rejY, { r: 3.2, color: pal.marker, label: "拒绝" }));
    if (ncX.length > 0) scene.layers.push(scatterLayer("ratio", ncX, ncY, { r: 4, color: pal.warn, label: "负曲率分支" }));
    scene.layers.push(
      lineLayer("ratio", { x: [rate.xlim[0], rate.xlim[1]], y: [0, 0], color: pal.reference, width: 1, dash: [3, 3] }),
    );
    if (t2 !== undefined) {
      scene.layers.push(
        lineLayer("ratio", { x: [rate.xlim[0], rate.xlim[1]], y: [t2, t2], color: pal.exact, width: 1.2, dash: [5, 3] }),
      );
    }
    scene.layers.push(
      lineLayer("ratio", { x: [rate.xlim[0], rate.xlim[1]], y: [0.25, 0.25], color: pal.curves[2] as string, width: 1.1, dash: [2, 4] }),
    );
  }

  const budget = unitBudget(L.rightW);
  if (L.rightW >= 200) {
    let y = L.topY;
    const textBottom = L.botY - 10;
    const push = (body: string, color: string): void => {
      if (y + TEXT_ROW_H > textBottom) return;
      const block = textBlock(L.rightX, y, textBottom, [body], color, budget, "top");
      scene.texts.push(...block);
      y += block.length * TEXT_ROW_H;
    };
    push(`k=${k}/${kMax}，Δ_k=${formatNumber(radiusNow ?? Number.NaN, 3)}（Δ0=${formatNumber(delta0 ?? Number.NaN, 3)}）。`, pal.ink);
    push(
      k === 0
        ? "ρ_当前：还没有执行任何一步（null，不是 0）。"
        : ratioNow === undefined
          ? "ρ_当前：帧没有给出（不是 0）。"
          : `ρ_当前=${formatNumber(ratioNow, 3)} → ${acceptedNow ? "接受" : "拒绝"}。`,
      pal.ink,
    );
    push(`累计：接受 ${formatNumber(nAccepted ?? Number.NaN, 2)}、拒绝 ${formatNumber(nRejected ?? Number.NaN, 2)} 步。`, pal.muted);
    push(`负曲率分支 ${formatNumber(nNegCurv ?? Number.NaN, 0)} 次；最终 ‖x−x*‖=${formatNumber(finalDist ?? Number.NaN, 3)}。`, pal.muted);
    push(`t2=${formatNumber(t2 ?? Number.NaN, 3)}，t1=0.25。`, pal.muted);
    push("ρ 可以为负（那步会让 f 上升）。", pal.muted);
    push("对数轴画不出负值，所以 ρ 单独用线性轴。", pal.muted);
    push("轨迹只含被接受的步；红点=被拒的尝试。", pal.muted);
    push("圆是显示辅助；半径读数看 Δ 轴。", pal.muted);
    push("窗口裁掉一段圆，不算数据溢出。", pal.muted);
    push(
      overflow ? "帧标记 overflow=true：轨迹冲出了固定窗口（见溢出角标）。" : "固定窗口 [-3.5,3.5]×[-1.5,12]。",
      overflow ? pal.warn : pal.muted,
    );
  } else {
    degraded.push("右侧说明列过窄：读数并入角标");
  }

  scene.badges.push(placeBadge(sourceBadge(dataSourceOf(frame)), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge(
        "status",
        `k=${k}｜Δ=${formatNumber(radiusNow ?? Number.NaN, 3)}｜ρ=${k === 0 || ratioNow === undefined ? "—" : formatNumber(ratioNow, 3)}｜接受 ${formatNumber(nAccepted ?? Number.NaN, 2)}/拒绝 ${formatNumber(nRejected ?? Number.NaN, 2)}`,
        8,
        8,
      ),
      SLOT.status,
      width,
    ),
  );
  scene.badges.push(
    placeBadge(badge("status", `终止：${stringOf(frame, "stop") ?? "?"}；轨迹只含接受步`, 8, 8), SLOT.aux, width),
  );
  if (overflow) {
    scene.overflow = { flag: true, note: "轨迹或当前点越出固定窗口；半径读数仍在副轴" };
    scene.badges.push(placeBadge(badge("overflow", "溢出：轨迹越出固定窗口，见状态角标", 8, 8), SLOT.overflow, width));
  }
  const dbg = degradedBadge(degraded, SLOT.degraded, width);
  if (dbg !== undefined) scene.badges.push(dbg);

  requireDrawableLayers(scene);
  return scene;
};
