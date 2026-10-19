/**
 * `convergence_rates`（5.1 章 viz 2）：收敛速度的证据链。
 *
 * 这张图**不看斜率陡不陡**，只看比值：
 * - 牛顿的二次比值 q_k = e_{k+1}/e_k^2 必须在若干步里**趋于常数**；
 * - GD 的线性比值 r_k = e_{k+1}/e_k 必须趋于理论值 1-αμ；
 * - 误差贴到舍入地板之后 q_k 会爆炸（实测到 3.35e122），这是**假超二次**，
 *   不是收敛变快；q 的显示窗口因此只到 1e10，越窗由溢出角标点名。
 *
 * q 与 r 的量级差 130 个数量级，硬塞进同一根对数轴会把 r 压成一条看不见的线；
 * 所以底部拆成两根轴：`ratio`（q，对数）与 `linear_rate`（r，线性 0.95–1）。
 *
 * 帧 `time.mode="none"`：横轴是迭代数，`scene.t` 恒为 0。
 */

import { formatNumber } from "../../shared/src/axes.js";
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
  stringOf,
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
import {
  TEXT_ROW_H,
  axisTitleOf,
  axisYlabelOf,
  columnsLayout,
  textBlock,
  unitBudget,
} from "../_common/panel.js";
import { CHAPTER, SLOT, methodColors, prefix } from "./common.js";
import { manifest } from "./manifest.gen.js";

const VIZ = "convergence_rates";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "收敛速度：q 趋于常数才是二次收敛";

const Q_FLOOR_FALLBACK = 1e-30;
const Q_CEIL_FALLBACK = 1e10;

export const convergenceRatesModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  const iterations = requireArray(frame, "iterations");
  const errors: Record<string, ArrayLike<number>> = {};
  for (const name of ["gd", "momentum", "bfgs", "newton"] as const) {
    errors[name] = requireArray(frame, `error_${name}`);
    requireSameLength(iterations, errors[name]!, `iterations 与 error_${name}`);
  }
  const q = requireArray(frame, "q_newton");
  const r = requireArray(frame, "r_gd");
  const qValid = requireArray(frame, "q_newton_valid");
  const rValid = requireArray(frame, "r_gd_valid");
  requireSameLength(q, qValid, "q_newton 与 q_newton_valid");
  requireSameLength(r, rValid, "r_gd 与 r_gd_valid");
  if (q.length > iterations.length - 1 || r.length > iterations.length - 1) {
    throw new Error("比值数组长度超过 iterations-1：每步比值的横轴没有定义");
  }
  const pitfalls = requirePitfalls(frame);

  const degraded: string[] = [];
  const floorMeta = metaNumber(frame, "error_floor");
  const errFloor = floorMeta !== undefined && floorMeta > 0 ? floorMeta : 1e-17;
  if (floorMeta === undefined) degraded.push("帧 meta 缺 error_floor：误差地板用 1e-17");
  const ratioFloorMeta = metaNumber(frame, "ratio_floor");
  const qFloor = ratioFloorMeta !== undefined && ratioFloorMeta > 0 ? ratioFloorMeta : Q_FLOOR_FALLBACK;
  if (ratioFloorMeta === undefined) degraded.push("帧 meta 缺 ratio_floor：比值地板用 1e-30");
  const qCeilMeta = metaNumber(frame, "ratio_display_ceiling");
  const qCeil = qCeilMeta !== undefined && qCeilMeta > qFloor ? qCeilMeta : Q_CEIL_FALLBACK;

  const enter = metaNumber(frame, "newton_enter_iter");
  const floorStart = metaNumber(frame, "newton_floor_start");
  const qConst = metaNumber(frame, "newton_quadratic_constant");
  const orderTail = metaNumber(frame, "newton_order_tail");
  const gdMeasured = metaNumber(frame, "gd_linear_rate_measured");
  const gdTheory = metaNumber(frame, "gd_linear_rate_theory");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: 0,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = columnsLayout(scene, width, height, { topShare: 0.46, gap: 50 });

  const kMax = Math.max(1, (iterations[iterations.length - 1] as number) ?? 1);
  const conv = makeAxis("convergence", [L.left, L.topY, L.leftW, L.topH], {
    xlim: [0, kMax],
    ylim: [errFloor, 3],
    yscale: "log",
    xlabel: "",
    ylabel: axisYlabelOf(SPEC?.axes, "convergence", "‖x_k - x*‖"),
    title: axisTitleOf(SPEC?.axes, "convergence", "到最优解的距离（对数轴）"),
    autoscaled: true,
  });

  // 底部左侧：q（对数，显示窗口 [qFloor, qCeil]）；右侧：r（线性 0.95–1）。
  // 窄画布上左列放不下两根并排轴：只保留 requiredAux 更相关的 q 轴，r 明说省略。
  const narrow = L.leftW < 320;
  const qW = narrow ? L.leftW : Math.max(140, Math.round(L.leftW * 0.6));
  const rX = L.left + qW + 54;
  const rW = narrow ? 0 : Math.max(90, L.leftW - qW - 54);
  // q 的显示窗只包住“有效比值 + q=1 + 实测二次常数”，不把 1e-30 地板拉进来：
  // 地板是“无定义点的占位”，不是 q 的取值；把它当轴下界会把真正的 q 压成一条线。
  let qValidLo = Number.POSITIVE_INFINITY;
  let qValidHi = 0;
  for (let i = 0; i < q.length; i += 1) {
    const v = q[i] as number;
    if ((qValid[i] as number) > 0.5 && Number.isFinite(v) && v > 0) {
      qValidLo = Math.min(qValidLo, v);
      qValidHi = Math.max(qValidHi, v);
    }
  }
  const qLo = Math.max(qFloor, (Number.isFinite(qValidLo) ? qValidLo : 1e-4) * 0.3);
  const qHi = Math.max(2, Math.min(qCeil, qValidHi > 0 ? Math.max(2, Math.min(10, qValidHi * 2)) : 2));
  // q 是牛顿专属的比值：有效段只有前几步，横轴不该硬拉到 GD 的 30 步。
  let qLast = 0;
  let qExit = -1;
  for (let i = 0; i < q.length; i += 1) {
    if ((qValid[i] as number) > 0.5 && Number.isFinite(q[i] as number)) {
      const at = (iterations[Math.min(i, iterations.length - 1)] as number) ?? i;
      qLast = Math.max(qLast, at);
      if (qExit < 0 && (q[i] as number) > qHi) qExit = at;
    }
  }
  // 窗口只画到 q 第一次越窗（之后的值是舍入噪声放大器，不是收敛证据）。
  const qXmax = Math.max(2, Math.min(kMax, (qExit >= 0 ? qExit : qLast) + 1));
  const ratio = makeAxis("ratio", [L.left, L.botY, qW, L.botH], {
    xlim: [0, qXmax],
    ylim: [qLo, qHi],
    yscale: "log",
    xlabel: "迭代步 k（越窗后截断）",
    ylabel: "q_k",
    title: "q_k：牛顿二次比值（对数）",
    autoscaled: true,
  });
  scene.axes.push(conv, ratio);
  let rate: ReturnType<typeof makeAxis> | undefined;
  if (!narrow) {
    rate = makeAxis("linear_rate", [rX, L.botY, rW, L.botH], {
      xlim: [0, kMax],
      ylim: [0.95, 1.0],
      xlabel: "迭代步 k",
      ylabel: "r_k",
      title: "GD 比值 r_k",
      autoscaled: false,
    });
    scene.axes.push(rate);
  } else {
    degraded.push("画布过窄：GD 线性比值 r 的独立轴省略（q 与收敛轴仍在，读数见角标）");
  }

  const colors = methodColors(pal);
  const convOrder = ["gd", "momentum", "bfgs", "newton"] as const;
  for (const name of convOrder) {
    scene.layers.push(
      lineLayer("convergence", {
        x: iterations,
        y: errors[name]!,
        color: colors[name],
        width: name === "newton" ? 2.2 : 1.7,
      }),
    );
  }
  scene.layers.push(
    lineLayer("convergence", {
      x: [conv.xlim[0], conv.xlim[1]],
      y: [errFloor, errFloor],
      color: pal.warn,
      width: 1,
      dash: [3, 3],
    }),
  );
  const marks: Array<{ x: number; dash: number[]; color: string }> = [];
  if (enter !== undefined && enter >= 0) marks.push({ x: enter, dash: [4, 3], color: pal.exact });
  if (floorStart !== undefined && floorStart >= 0) marks.push({ x: floorStart, dash: [2, 4], color: pal.warn });
  for (const m of marks) {
    if (m.x < conv.xlim[0] || m.x > conv.xlim[1]) continue;
    scene.layers.push(
      lineLayer("convergence", {
        x: [m.x, m.x],
        y: [conv.ylim[0], conv.ylim[1]],
        color: m.color,
        width: 1,
        dash: m.dash,
      }),
    );
    if (m.x <= qXmax) {
      scene.layers.push(
        lineLayer("ratio", {
          x: [m.x, m.x],
          y: [ratio.ylim[0], ratio.ylim[1]],
          color: m.color,
          width: 1,
          dash: m.dash,
        }),
      );
    }
  }

  // q：只连有效且相邻的点。无效位置是“比值无定义”，不是 0，也不是“曲线断在别处”。
  const qx = prefix(iterations, q.length);
  let segment: { x: number[]; y: number[] } = { x: [], y: [] };
  const segments: Array<{ x: number[]; y: number[] }> = [];
  const qInvalidX: number[] = [];
  for (let i = 0; i < q.length; i += 1) {
    const ok = (qValid[i] as number) > 0.5 && Number.isFinite(q[i] as number);
    // 横轴已声明“越窗后截断”：x 超过显示窗的点直接不进折线，否则 x 越窗会再报一枚
    // 与轴名矛盾（图上明明裁掉了）的溢出角标。q 值本身超出显示上限仍由 y 越窗点名。
    if (ok && (qx[i] as number) <= qXmax) {
      segment.x.push(qx[i] as number);
      segment.y.push(q[i] as number);
    } else {
      if (segment.x.length > 1) segments.push(segment);
      segment = { x: [], y: [] };
      if (!ok) qInvalidX.push(qx[i] as number);
    }
  }
  if (segment.x.length > 1) segments.push(segment);
  for (const s of segments) {
    scene.layers.push(
      lineLayer("ratio", {
        x: s.x,
        y: s.y,
        color: pal.residual,
        width: 1.8,
      }),
    );
  }
  const qInvalidIn = qInvalidX.filter((x) => x <= qXmax);
  if (qInvalidIn.length > 0) {
    // 窗内的无定义点画在轴底并明确写“不是 q 值”；窗外的在图注里报总数，绝不静默。
    scene.layers.push(
      scatterLayer("ratio", qInvalidIn, qInvalidIn.map(() => qLo), {
        r: 2.4,
        color: pal.muted,
        label: "q 无定义（掩码=0）",
      }),
    );
  }
  scene.layers.push(
    lineLayer("ratio", {
      x: [ratio.xlim[0], ratio.xlim[1]],
      y: [1, 1],
      color: pal.warn,
      width: 1,
      dash: [2, 4],
    }),
  );
  if (qConst !== undefined && enter !== undefined && floorStart !== undefined && floorStart > enter) {
    scene.layers.push(
      lineLayer("ratio", {
        x: [enter, floorStart],
        y: [qConst, qConst],
        color: pal.exact,
        width: 1.4,
        dash: [5, 3],
      }),
    );
  }

  // r：线性轴，只有 [0.95,1] 这一小段；理论线与实测同屏。
  const rx = prefix(iterations, r.length);
  const rValidPts: { x: number[]; y: number[] } = { x: [], y: [] };
  const rBad: number[] = [];
  for (let i = 0; i < r.length; i += 1) {
    if ((rValid[i] as number) > 0.5 && Number.isFinite(r[i] as number)) {
      rValidPts.x.push(rx[i] as number);
      rValidPts.y.push(r[i] as number);
    } else {
      rBad.push(rx[i] as number);
    }
  }
  if (rValidPts.x.length >= 2) {
    if (rate !== undefined) scene.layers.push(lineLayer("linear_rate", { x: rValidPts.x, y: rValidPts.y, color: colors.gd, width: 1.9 }));
  }
  if (rate !== undefined && rBad.length > 0) {
    scene.layers.push(
      scatterLayer("linear_rate", rBad, rBad.map(() => rate.ylim[0]), { r: 2.2, color: pal.muted, label: "r 无定义" }),
    );
  }
  if (rate !== undefined && gdTheory !== undefined && gdTheory >= rate.ylim[0] && gdTheory <= rate.ylim[1]) {
    scene.layers.push(
      lineLayer("linear_rate", {
        x: [rate.xlim[0], rate.xlim[1]],
        y: [gdTheory, gdTheory],
        color: pal.exact,
        width: 1.3,
        dash: [5, 3],
      }),
    );
  }

  // 侧栏：把“读出来的数”与“怎么读”分开。
  const budget = unitBudget(L.rightW);
  const textBottom = L.botY - 10;
  if (L.rightW >= 200) {
    let y = L.topY;
    const push = (body: string, color: string): void => {
      if (y + TEXT_ROW_H > textBottom) return;
      const block = textBlock(L.rightX, y, textBottom, [body], color, budget, "top");
      scene.texts.push(...block);
      y += block.length * TEXT_ROW_H;
    };
    push(`牛顿：q 常数 ${formatNumber(qConst ?? Number.NaN, 3)} @ k≥${formatNumber(enter ?? Number.NaN, 0)}，尾部阶 ${formatNumber(orderTail ?? Number.NaN, 3)}（≈2 才是二次）。`, pal.ink);
    push(`GD：实测 r=${formatNumber(gdMeasured ?? Number.NaN, 4)}，理论 ${formatNumber(gdTheory ?? Number.NaN, 4)}；地板从 k=${formatNumber(floorStart ?? Number.NaN, 0)} 起污染 q。`, pal.muted);
    push(
      `q 无定义 ${qInvalidX.length} 个（灰点画在轴底，不是 q=0）；越窗由溢出角标点名，不静默裁剪。`,
      qInvalidX.length > 0 ? pal.warn : pal.muted,
    );
    push("颜色：GD 蓝、动量绿、BFGS 黄、牛顿紫。", pal.muted);
    push("橙竖线=进入二次区；红竖线=地板开始污染 q；橙短横=实测二次常数。", pal.muted);
    push(`终止：牛顿 ${stringOf(frame, "stop_newton") ?? "?"}，GD ${stringOf(frame, "stop_gd") ?? "?"}，BFGS ${stringOf(frame, "stop_bfgs") ?? "?"}。`, pal.muted);
  } else {
    degraded.push("右侧说明列过窄：读数并入角标，口径见 pitfalls");
  }

  // 状态角标：只放最关键的三个数。
  scene.badges.push(placeBadge(sourceBadge(dataSourceOf(frame)), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(true), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge(
        "status",
        `q 常数 ${formatNumber(qConst ?? Number.NaN, 3)}｜尾部阶 ${formatNumber(orderTail ?? Number.NaN, 2)}｜r=${formatNumber(gdMeasured ?? Number.NaN, 4)}`,
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
