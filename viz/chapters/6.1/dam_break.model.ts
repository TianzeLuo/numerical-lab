/**
 * `dam_break`（6.1 章 viz 4）：干底溃坝 vs Ritter 精确解。
 *
 * 三条纪律
 * -------
 * 1. **窗口决定误差**：Ritter 解只在扇区 −c_L ≤ ξ ≤ 2c_L 内光滑；全域 L1 多出来的
 *    主要是干锋位置误差。侧栏把扇内 / 全域两个数并排报，不把“全域更差”说成格式精度。
 * 2. **干底质量漂移不是不守恒**：正性地板每步注入 min_depth·Δx 的水；帧里的
 *    mass_drift 是这条注入的账，不是通量散度。
 * 3. **误差掩码**：`error_valid=0` 的点是地板占位，不画成“误差恰好 1e-14”；
 *    t=0 时没有推进步，误差面板为空并如实说明。
 *
 * 帧 `time.mode="scrub"`：`t` 是物理时间。
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
  bandLayer,
  lineLayer,
  makeAxis,
  scatterLayer,
  sourceBadge,
} from "../../shared/src/scene.js";
import { axisTitleOf, axisYlabelOf, columnsLayout } from "../_common/panel.js";
import { CHAPTER, SLOT, extentOf, fmt, pushGroups, singleColumn as singleCol, statusLine } from "./common.js";
import { manifest } from "./manifest.gen.js";

const VIZ = "dam_break";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "干底溃坝：Ritter 精确解、误差窗口与 CFL";

export const damBreakModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  const x = requireArray(frame, "x");
  const h = requireArray(frame, "h");
  const hu = requireArray(frame, "hu");
  const u = requireArray(frame, "u");
  const hExact = requireArray(frame, "h_exact");
  const uExact = requireArray(frame, "u_exact");
  const hInitial = requireArray(frame, "h_initial");
  const error = requireArray(frame, "error");
  const errorValid = requireArray(frame, "error_valid");
  const fan = requireArray(frame, "fan");
  requireSameLength(x, h, "x 与 h");
  requireSameLength(x, hExact, "x 与 h_exact");
  requireSameLength(x, hInitial, "x 与 h_initial");
  requireSameLength(x, error, "x 与 error");
  requireSameLength(x, errorValid, "x 与 error_valid");
  requireSameLength(x, fan, "x 与 fan");
  if (hu.length !== h.length || u.length !== h.length || uExact.length !== h.length) {
    throw new Error("dam_break 的 h/hu/u/u_exact 长度不一致");
  }
  const pitfalls = requirePitfalls(frame);

  const degraded: string[] = [];
  const t = metaNumber(frame, "t") ?? opts.t;
  const n = metaNumber(frame, "n");
  const cfl = metaNumber(frame, "cfl");
  const maxCfl = metaNumber(frame, "max_cfl");
  const nSteps = metaNumber(frame, "n_steps");
  const dt = metaNumber(frame, "dt");
  const l1Fan = metaNumber(frame, "l1_fan");
  const l1All = metaNumber(frame, "l1_all");
  const l1Ratio = metaNumber(frame, "l1_ratio");
  const massDrift = metaNumber(frame, "mass_drift");
  const mass = metaNumber(frame, "mass");
  const mass0 = metaNumber(frame, "mass0");
  const hMin = metaNumber(frame, "h_min");
  const energy = metaNumber(frame, "energy");
  const fanLo = metaNumber(frame, "fan_lo");
  const fanHi = metaNumber(frame, "fan_hi");
  const reached = (metaNumber(frame, "reached") ?? 1) > 0.5;
  const finite = (metaNumber(frame, "finite") ?? 1) > 0.5;
  const status = (frame.strings["status"] as string | undefined) ?? "?";

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const wide = width >= 820;
  const L = wide ? columnsLayout(scene, width, height, { topShare: 0.62, auxH: 140, gap: 56 }) : null;
  const S = wide ? null : singleCol(scene, width, height, 0.58);
  const topY = L ? L.topY : (S as NonNullable<typeof S>).topY;
  const floorY = L ? L.floor : (S as NonNullable<typeof S>).floor;
  const mainRect: [number, number, number, number] = L ? [L.left, L.topY, L.leftW, L.topH] : (S as NonNullable<typeof S>).main;
  const errRect: [number, number, number, number] = L ? [L.left, L.botY, L.leftW, L.botH] : (S as NonNullable<typeof S>).aux;

  const hRange = extentOf([h, hExact, hInitial], [0, 1]);
  const yMax = Math.max(0.1, hRange[1] * 1.12);
  const main = makeAxis("main", mainRect, {
    xlim: [0, 1],
    ylim: [0, yMax],
    xlabel: "$x$",
    ylabel: axisYlabelOf(SPEC?.axes, "main", "h"),
    title: axisTitleOf(SPEC?.axes, "main", "水深 h(x) 与 Ritter 精确解"),
    autoscaled: false,
    tickTarget: 4,
  });

  // 逐点误差只在掩码为真的点上取值；占位值（1e-14）绝不能当数据画进对数轴。
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < error.length; i += 1) {
    const v = error[i] as number;
    if ((errorValid[i] as number) <= 0.5 || !Number.isFinite(v) || v <= 0) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const hasError = Number.isFinite(lo) && Number.isFinite(hi) && hi > 0;
  const yLo = hasError ? Math.max(1e-15, lo * 0.5) : 1e-7;
  const yHi = hasError ? Math.max(yLo * 20, hi * 2.2) : 1e-5;
  if (!hasError) degraded.push("本帧没有有效的逐点误差（t=0 或全部贴地板）：误差面板为空");
  const errAxis = makeAxis("error", errRect, {
    xlim: [0, 1],
    ylim: [yLo, yHi],
    yscale: "log",
    xlabel: "$x$",
    ylabel: axisYlabelOf(SPEC?.axes, "error", "绝对误差"),
    title: axisTitleOf(SPEC?.axes, "error", "|h_num − h_exact|（逐点，对数；掩码外的点不画）"),
    autoscaled: true,
    tickTarget: 4,
  });
  scene.axes.push(main, errAxis);

  if (fanLo !== undefined && fanHi !== undefined && fanHi > fanLo) {
    scene.layers.push(
      bandLayer("main", {
        x: [fanLo, fanHi],
        ylo: [0, 0],
        yhi: [yMax, yMax],
        color: pal.muted,
        alpha: 0.12,
        label: "Ritter 扇区",
      }),
    );
  }
  scene.layers.push(
    lineLayer("main", { x, y: hInitial, color: pal.reference, width: 0.9, dash: [3, 3], label: "初始水深 $h_0$" }),
    lineLayer("main", { x, y: hExact, color: pal.exact, width: 1.4, dash: [6, 3], label: "Ritter 精确解" }),
    lineLayer("main", { x, y: h, color: pal.curves[0] as string, width: 2, label: "数值解" }),
  );

  if (hasError) {
    const xe: number[] = [];
    const ye: number[] = [];
    for (let i = 0; i < error.length; i += 1) {
      const v = error[i] as number;
      if ((errorValid[i] as number) <= 0.5 || !Number.isFinite(v) || v <= 0) continue;
      xe.push(x[i] as number);
      ye.push(v);
    }
    scene.layers.push(scatterLayer("error", xe, ye, { r: 1.7, color: pal.residual, label: "逐点误差" }));
  }

  if (L && L.rightW >= 200) {
    const uMax = extentOf([u], [0, 1])[1];
    pushGroups(
      scene,
      L.rightX,
      topY,
      floorY,
      L.rightW,
      [
        {
          lines: [
            `t=${fmt(t, 4)}，n=${fmt(n, 0)}，cfl=${fmt(cfl, 2)}。`,
            `${fmt(nSteps, 0)} 步；max CFL ${fmt(maxCfl, 3)}；u_max=${fmt(uMax, 3)}。`,
            `dt=${fmt(dt, 2)}；${reached ? "已到目标时刻。" : "未到目标时刻。"}`,
          ],
          color: pal.ink,
        },
        {
          lines: t <= 0
            ? [
                "t=0：尚未推进，无扇区。",
                "扇内 L1=0；全域 L1=0。",
                "这不是“误差为零”。",
              ]
            : [
                `扇内 L1=${fmt(l1Fan, 2)}。`,
                `全域 L1=${fmt(l1All, 2)}（比值 ${fmt(l1Ratio, 2)}）。`,
                "扇内才是干净估计。",
                "全域多的是干锋位置误差。",
              ],
          color: pal.muted,
        },
        {
          lines: [
            `质量 ${fmt(mass, 6)}（初值 ${fmt(mass0, 6)}）。`,
            `漂移 ${fmt(massDrift, 2)}；地板注入水，非不守恒。`,
            `h_min=${fmt(hMin, 2)}；能量 ${fmt(energy, 4)}。`,
          ],
          color: pal.muted,
        },
        {
          lines: [
            "蓝=h_num；橙虚=Ritter。",
            "灰虚=h_initial；灰带=扇区。",
            hasError ? "紫点=逐点误差（只画有效点）。" : "误差面板：无有效点。",
            `状态 ${status}。${finite ? "" : "出现非有限值。"}`,
          ],
          color: finite ? pal.muted : pal.warn,
        },
      ],
      pal,
    );
  } else if (L) {
    degraded.push("右侧说明列过窄：读数并入角标");
  } else {
    degraded.push("窄画布：省略右栏说明");
  }

  scene.badges.push(placeBadge(sourceBadge(dataSourceOf(frame)), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge(
        "status",
        statusLine([
          `n=${fmt(n, 0)}`,
          `cfl ${fmt(cfl, 1)}→${fmt(maxCfl, 2)}`,
          `L1 ${fmt(l1Fan, 2)}`,
          `${fmt(nSteps, 0)} 步`,
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
