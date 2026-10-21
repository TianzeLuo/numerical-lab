/**
 * `packet_long_time`（4.2 章）的 model：长时波包把「耗散」与「色散」画成两张不同的脸。
 *
 * 图层（顺序即绘制顺序）
 * ----------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | band | `packet` | 前缘取样区 x<0.25（浅蓝） |
 * | 2 | band | `packet` | 拖尾取样区 x>0.75（浅红） |
 * | 3 | line | `packet` | 初值（灰虚线） |
 * | 4 | line | `packet` | 精确平移解（橙） |
 * | 5 | line | `packet` | 数值解（蓝，粗） |
 * | 6 | line | `packet` | 精确包络（灰点线） |
 * | 7 | line | `packet` | 数值包络（绿，粗） |
 * | 8 | line+scatter | `packet` | **幅值标尺**：同一 x 上精确包络与数值包络的竖直落差 |
 * | 9 | line | `error` | 逐点误差（对数，固定 1e-17–3） |
 * | 10 | scatter | `error` | 已到地板的点 |
 * | 11 | line | `residual` | RK 回代残差 / 积分器之差（对数，自适应） |
 *
 * 为什么要有「幅值标尺」：只看两条曲线，读者很难判断「包络矮了多少」是耗散还是
 * 仅仅相位错位。标尺在**同一个 x** 上量峰值落差，并把 meta 的三个归一化比值写进正文；
 * 拖尾/前缘则用固定的取样区带显示（`tail_ratio`/`front_ratio` 的定义就是这两块区域）。
 *
 * 固定窗口：波包 −0.3–1.3、误差 1e-17–3（manifest 声明 `autoscaled=false`）。
 * 越窗与贴地板都在正文与溢出角标里点名，绝不用自适应掩盖幅度差。
 *
 * 时间轴：`scrub`，`t` 是**域长的倍数**（`meta.periods_done`），不是秒。
 */

import { formatNumber } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import {
  applyOverflow,
  canvasSize,
  clamp,
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
  type SceneModel,
} from "../../shared/src/model.js";
import {
  badge,
  criticalBadge,
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
  maskIndices,
  positiveMinMax,
  sliceRun,
  textBlock,
  unitBudget,
} from "../_common/panel.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "4.2";
const VIZ = "packet_long_time";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "长时波包传播：耗散收缩与色散拖尾";
const T_MAX = SPEC && SPEC.time.mode !== "none" ? SPEC.time.tMax ?? 4 : 4;

const Y_PACKET: [number, number] = [-0.3, 1.3];
const Y_ERROR: [number, number] = [1e-17, 3];
const FLOOR_FALLBACK = 1e-17;

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

export const packetLongTimeModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组 ——
  const x = requireArray(frame, "x");
  const u = requireArray(frame, "u");
  const exact = requireArray(frame, "u_exact");
  const initial = requireArray(frame, "u_initial");
  const envNum = requireArray(frame, "envelope_num");
  const envExact = requireArray(frame, "envelope_exact");
  const errCurve = requireArray(frame, "error_curve");
  const residual = requireArray(frame, "residual_profile");
  const gap = requireArray(frame, "integrator_gap");
  for (const [name, arr] of [
    ["u", u],
    ["u_exact", exact],
    ["u_initial", initial],
    ["envelope_num", envNum],
    ["envelope_exact", envExact],
    ["error_curve", errCurve],
    ["residual_profile", residual],
    ["integrator_gap", gap],
  ] as const) {
    requireSameLength(x, arr, `帧数组 x 与 ${name}`);
  }
  const pitfalls = requirePitfalls(frame);

  // —— 可选数组 ——
  const degraded: string[] = [];
  const tailMask = frame.has("tail_region") ? requireArray(frame, "tail_region") : undefined;
  if (tailMask === undefined) degraded.push("帧缺 tail_region：不画拖尾取样区（不按 x>0.75 自己猜）");
  else requireSameLength(x, tailMask, "帧数组 x 与 tail_region");
  const frontMask = frame.has("front_region") ? requireArray(frame, "front_region") : undefined;
  if (frontMask === undefined) degraded.push("帧缺 front_region：不画前缘取样区（不按 x<0.25 自己猜）");
  else requireSameLength(x, frontMask, "帧数组 x 与 front_region");
  const atFloor = frame.has("error_at_floor") ? requireArray(frame, "error_at_floor") : undefined;
  if (atFloor === undefined) degraded.push("帧缺 error_at_floor：不标贴地板的点（地板仍画）");
  else requireSameLength(x, atFloor, "帧数组 x 与 error_at_floor");
  const uClipped = frame.has("u_clipped") ? requireArray(frame, "u_clipped") : undefined;
  if (uClipped === undefined) degraded.push("帧缺 u_clipped：不统计截断点数");

  // —— meta ——
  const periodsDoneMeta = metaNumber(frame, "periods_done");
  const tActualMeta = metaNumber(frame, "t_actual");
  const tActual = periodsDoneMeta ?? tActualMeta ?? clamp(opts.t, 0, T_MAX);
  if (periodsDoneMeta === undefined) degraded.push("帧 meta 缺 periods_done：时间按 t_actual/请求值显示");
  const tRequested = metaNumber(frame, "t_requested") ?? opts.t;
  const periodsReq = metaNumber(frame, "periods_requested");
  const nu = metaNumber(frame, "nu");
  if (nu === undefined) degraded.push("帧 meta 缺 nu：不显示 CFL 数（不按请求参数猜）");
  const nx = metaNumber(frame, "nx");
  if (nx === undefined) degraded.push("帧 meta 缺 nx：不显示格点数");
  const amp = metaNumber(frame, "amplitude_ratio");
  if (amp === undefined) degraded.push("帧 meta 缺 amplitude_ratio：不报幅值比（不自己算峰值）");
  const tail = metaNumber(frame, "tail_ratio");
  if (tail === undefined) degraded.push("帧 meta 缺 tail_ratio：不报拖尾比");
  const front = metaNumber(frame, "front_ratio");
  if (front === undefined) degraded.push("帧 meta 缺 front_ratio：不报前缘比");
  const l2 = metaNumber(frame, "l2_error");
  const umax = metaNumber(frame, "umax");
  const tv = metaNumber(frame, "total_variation");
  const maxG = metaNumber(frame, "max_abs_symbol");
  const decayOrder = metaNumber(frame, "decay_order");
  const dissMeta = frame.meta["dissipative"];
  const dispMeta = frame.meta["dispersive"];
  if (typeof dissMeta !== "boolean" || typeof dispMeta !== "boolean") {
    degraded.push("帧 meta 缺 dissipative/dispersive：不判耗散型/色散型（不按阈值自己分类）");
  }
  const isDiss = dissMeta === true;
  const isDisp = dispMeta === true;
  const scheme = stringOf(frame, "scheme");
  const limiter = stringOf(frame, "limiter");
  const floorText = stringOf(frame, "floor");
  const floor = numberOr(floorText, FLOOR_FALLBACK);
  if (floorText === undefined) degraded.push("帧 strings 缺 floor：地板标记按 1e-17");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tActual,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = columnsLayout(scene, width, height, { topShare: 0.6, auxH: 118, gap: 46 });
  const narrow = !L.columns;
  const panels = narrow ? narrowStack(L.floor, L.topY, width, 3) : undefined;

  const packet = makeAxis(
    "packet",
    panels ? panels[0] : [L.left, L.topY, L.leftW, L.topH],
    {
      xlim: [0, 1],
      ylim: Y_PACKET,
      xticks: [],
      xlabel: "",
      title: axisTitleOf(SPEC?.axes, "packet", "波包与精确解（固定窗口）"),
      ylabel: axisYlabelOf(SPEC?.axes, "packet", "u(x,t)"),
      autoscaled: false,
    },
  );
  const error = makeAxis(
    "error",
    panels ? panels[1] : [L.left, L.botY, L.leftW, L.botH],
    {
      xlim: [0, 1],
      ylim: Y_ERROR,
      yscale: "log",
      xticks: narrow ? [] : [0, 0.25, 0.5, 0.75, 1],
      xlabel: narrow ? "" : "x（格点，域长 L=1）",
      title: axisTitleOf(SPEC?.axes, "error", "逐点误差（对数纵轴）"),
      ylabel: axisYlabelOf(SPEC?.axes, "error", "|u − u_exact|"),
      autoscaled: false,
    },
  );
  const resLo = positiveMinMax(residual)?.[0] ?? floor;
  const gapLo = positiveMinMax(gap)?.[0] ?? floor;
  const resHi = Math.max(positiveMinMax(residual)?.[1] ?? 1, positiveMinMax(gap)?.[1] ?? 1);
  const residualAxis = makeAxis(
    "residual",
    panels ? panels[2] : [L.rightX, L.topY, L.rightW, L.auxH],
    {
      xlim: [0, 1],
      ylim: logRange(Math.min(resLo, gapLo) * 0.5, resHi * 2),
      yscale: "log",
      xticks: narrow ? [] : [0, 0.25, 0.5, 0.75, 1],
      xlabel: narrow ? "" : "x",
      title: "残差：RK 回代 − 积分器（对数）",
      ylabel: "$|r|$",
      autoscaled: true,
    },
  );
  if (narrow) {
    error.title = "";
    residualAxis.title = "";
  }
  scene.axes.push(packet, error, residualAxis);

  // —— 取样区带：先画（在曲线之下）——
  if (frontMask !== undefined) pushRegionBands(scene, "packet", x, frontMask, Y_PACKET, pal.note, 0.07);
  if (tailMask !== undefined) pushRegionBands(scene, "packet", x, tailMask, Y_PACKET, pal.warn, 0.07);

  // —— 主面板 ——
  scene.layers.push(
    lineLayer("packet", series(x, initial, pal.reference, { width: 1.2, dash: [4, 4], label: "初值" })),
    lineLayer("packet", series(x, exact, pal.exact, { width: 2, label: "精确解" })),
    lineLayer("packet", series(x, u, pal.curves[0], { width: 2.4, label: "数值解" })),
    lineLayer("packet", series(x, envExact, pal.reference, { width: 1.2, dash: [2, 3], label: "精确包络" })),
    lineLayer("packet", series(x, envNum, pal.curves[1], { width: 2, label: "数值包络" })),
  );

  // —— 幅值标尺：同一 x 上两条包络的落差 ——
  const peak = argmaxFinite(envNum);
  if (peak >= 0) {
    const xp = x[peak] as number;
    const yn = envNum[peak] as number;
    const ye = envExact[peak] as number;
    if (Number.isFinite(xp) && Number.isFinite(yn) && Number.isFinite(ye) && Math.abs(yn - ye) > 1e-9) {
      scene.layers.push(
        lineLayer("packet", series([xp, xp], [Math.min(yn, ye), Math.max(yn, ye)], pal.warn, {
          width: 1.4,
          dash: [3, 3],
        })),
        scatterLayer("packet", [xp, xp], [yn, ye], { r: 3, color: pal.warn }),
      );
    }
  }

  // —— 误差面板 ——
  scene.layers.push(
    lineLayer("error", series(x, errCurve, pal.curves[0], { width: 2, label: "误差 $|u − u_exact|$" })),
    lineLayer("error", series([0, 1], [floor, floor], pal.reference, { width: 1, dash: [2, 3], label: "舍入地板" })),
  );
  if (atFloor !== undefined) {
    const idx = maskIndices(atFloor);
    if (idx.length > 0) {
      const xs = new Float64Array(idx.length);
      const ys = new Float64Array(idx.length);
      idx.forEach((i, k) => {
        xs[k] = x[i] as number;
        ys[k] = floor;
      });
      scene.layers.push(scatterLayer("error", xs, ys, { r: 2.4, color: pal.warn, alpha: 0.85 }));
    }
  }

  // —— 残差副轴 ——
  scene.layers.push(
    lineLayer("residual", series(x, residual, pal.residual, { width: 1.8, label: "代数残差" })),
    lineLayer("residual", series(x, gap, pal.reference, { width: 1.2, dash: [5, 4], label: "积分器之差" })),
  );

  if (!narrow) {
    scene.texts.push(
      ...legendBlock(L.left + 6, L.topY + 6, L.leftW, "橙=精确　灰虚=初值　蓝=数值　绿=数值包络", pal.muted),
      ...legendBlock(
        L.left + 6,
        L.topY + 22,
        L.leftW,
        "竖虚线=同一 x 上包络峰值的落差（耗散）；淡色带=前缘/拖尾固定取样区",
        pal.muted,
      ),
      ...legendBlock(L.left + 6, L.botY + 6, L.leftW, "对数窗 1e-17–3；贴地板=舍入级，不是 0", pal.muted),
      ...legendBlock(L.rightX + 6, L.topY + 6, L.rightW, "紫=回代残差　灰虚=积分器之差", pal.muted),
      ...legendBlock(L.rightX + 6, L.topY + 38, L.rightW, "对数：贴地板=舍入级，不是 0", pal.muted),
    );
  }

  // —— 右列文字 ——
  const nFloor = atFloor !== undefined ? countMask(atFloor) : 0;
  const nClip = uClipped !== undefined ? countMask(uClipped) : 0;
  const nAbove = countAbove(errCurve, Y_ERROR[1]);
  if (!narrow) {
    const head = paramSummary(frame, opts, [
      { name: "nu", label: "$ν$" },
      { name: "nx", label: "$nx$" },
    ]);
    degraded.push(...head.degraded);
    const lines: string[] = [];
    lines.push(`${scheme ?? "?"}${limiter !== undefined && limiter !== "none" ? `(${limiter})` : ""}　${head.text}`);
    lines.push(`t=${formatNumber(tActual, 3)} 个域长（请求 ${formatNumber(periodsReq ?? tRequested, 3)}）`);
    if (amp !== undefined) lines.push(`幅值比 ${formatNumber(amp, 4)}（末态/初值峰值）`);
    if (front !== undefined) lines.push(`前缘比 ${formatNumber(front, 4)}（x<0.25）`);
    if (tail !== undefined) lines.push(`拖尾比 ${formatNumber(tail, 4)}（x>0.75）`);
    if (typeof dissMeta === "boolean" && typeof dispMeta === "boolean") {
      lines.push(
        isDiss && isDisp
          ? "帧判据：耗散与色散同时成立"
          : isDiss
            ? "帧判据：耗散型（包络压矮、尾巴小）"
            : isDisp
              ? "帧判据：色散型（包络保持、拖振）"
              : amp !== undefined && tail !== undefined && amp < 0.9 && tail >= 0.15
              ? "帧判据未命中：幅值<0.9 且拖尾≥0.15"
              : "两布尔都未命中（阈值口径见 pitfalls）",
      );
    }
    if (decayOrder !== undefined) lines.push(`前导耗散阶 φ^${formatNumber(decayOrder, 0)}（d2=0≠无耗散）`);
    lines.push("取样窗固定；periods 大时主包绕回，tail 不再单调");
    if (nClip > 0) lines.push(`波包 ${nClip}/${x.length} 点被截断（|u|>100）`);
    if (nAbove > 0) lines.push(`误差 ${nAbove} 点 >3：越窗（不是 3）`);
    if (nFloor > 0) lines.push(`${nFloor}/${x.length} 点贴 ${formatNumber(floor, 3)} 地板：舍入级`);
    if (l2 !== undefined) lines.push(`L2 误差 ${formatNumber(l2, 4)}`);
    if (umax !== undefined) lines.push(`umax ${formatNumber(umax, 4)}`);
    if (tv !== undefined) lines.push(`TV ${formatNumber(tv, 4)}`);
    if (maxG !== undefined) lines.push(`max|G| ${formatNumber(maxG, 5)}`);
    scene.texts.push(...textBlock(L.rightX, L.textTop, L.floor, lines, pal.muted, unitBudget(L.rightW), "top"));
  } else {
    degraded.push("窄画布（单列）：右列说明与图例未排下，波包窗仍固定");
  }

  // —— 角标 ——
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(
    placeBadge(badge("autoscale", "波包/误差固定；残差副轴自适应（对数）", 8, 26), SLOT.axisMode, width),
  );
  scene.badges.push(
    placeBadge(badge("autoscale", "波包窗 −0.3–1.3；误差窗 1e-17–3（不随帧变）", 8, 44), SLOT.auxAxis, width),
  );
  if (nu !== undefined) {
    scene.badges.push(
      placeBadge(criticalBadge("ν", nu, [criticalValue(frame, 1.0)], { beyond: "above" }), SLOT.critical, width),
    );
  }
  const status: string[] = [`t=${formatNumber(tActual, 2)}L`];
  if (amp !== undefined) status.push(`amp ${formatNumber(amp, 3)}`);
  if (tail !== undefined) status.push(`tail ${formatNumber(tail, 3)}`);
  status.push(
    typeof dissMeta === "boolean" && typeof dispMeta === "boolean"
      ? isDiss
        ? "耗散"
        : isDisp
          ? "色散"
          : "判据未命中"
      : "判据未给",
  );
  scene.badges.push(placeBadge(badge("status", status.join("；"), 8, 8), SLOT.status, width));
  const degradedNode = degradedBadge(degraded, SLOT.degraded, width);
  if (degradedNode !== undefined) scene.badges.push(degradedNode);

  requireDrawableLayers(scene);
  const nBad =
    countNonFinite(x) +
    countNonFinite(u) +
    countNonFinite(errCurve) +
    countNonFinite(envNum) +
    countNonFinite(residual) +
    (atFloor !== undefined ? countNonFinite(atFloor) : 0) +
    (uClipped !== undefined ? countNonFinite(uClipped) : 0) +
    (tailMask !== undefined ? countNonFinite(tailMask) : 0) +
    (frontMask !== undefined ? countNonFinite(frontMask) : 0);
  applyOverflow(scene, nBad, { slot: SLOT.overflow, width });
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具（显示几何与计数，不算物理量）
// --------------------------------------------------------------------------- //

function narrowStack(
  floor: number,
  top: number,
  width: number,
  n: number,
): [number, number, number, number][] {
  const gap = 12;
  const avail = Math.max(40, floor - top);
  const h = Math.max(20, Math.floor((avail - gap * (n - 1)) / n));
  const w = Math.max(120, width - 88);
  const out: [number, number, number, number][] = [];
  for (let i = 0; i < n; i += 1) out.push([64, top + i * (h + gap), w, h]);
  return out;
}

/** 把掩码为 1 的连续段画成竖直带（取样区是固定的，带的位置也固定）。 */
function pushRegionBands(
  scene: ReturnType<SceneModel>,
  axis: string,
  x: ArrayLike<number>,
  mask: ArrayLike<number>,
  y: readonly [number, number],
  color: string,
  alpha: number,
): void {
  let start = -1;
  const push = (end: number): void => {
    if (start < 0 || end - start < 1) return;
    const seg = sliceRun(x, x, { start, end });
    const ylo = new Float64Array(seg.x.length).fill(y[0]);
    const yhi = new Float64Array(seg.x.length).fill(y[1]);
    scene.layers.push({
      kind: "band",
      axes: axis,
      band: { x: seg.x, ylo, yhi, color, alpha },
    });
  };
  for (let i = 0; i < mask.length; i += 1) {
    const on = (mask[i] ?? 0) > 0.5;
    if (on && start < 0) start = i;
    if (!on && start >= 0) {
      push(i);
      start = -1;
    }
  }
  push(mask.length);
}

function countMask(a: ArrayLike<number>): number {
  let n = 0;
  for (let i = 0; i < a.length; i += 1) if ((a[i] ?? 0) > 0.5) n += 1;
  return n;
}

function countAbove(a: ArrayLike<number>, limit: number): number {
  let n = 0;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (Number.isFinite(v) && v > limit) n += 1;
  }
  return n;
}

function argmaxFinite(a: ArrayLike<number>): number {
  let best = -1;
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < a.length; i += 1) {
    const v = a[i];
    if (Number.isFinite(v) && v > hi) {
      hi = v;
      best = i;
    }
  }
  return best;
}

function criticalValue(frame: Parameters<SceneModel>[0], fallback: number): number {
  const raw = frame.critical["nu"];
  return typeof raw === "number" && Number.isFinite(raw) ? raw : fallback;
}

function numberOr(text: string | undefined, fallback: number): number {
  if (text === undefined) return fallback;
  const v = Number(text);
  return Number.isFinite(v) && v > 0 ? v : fallback;
}
