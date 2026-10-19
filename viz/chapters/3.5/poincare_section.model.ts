/**
 * `poincare_section`（3.5 章）的 model：三维轨道 + 截面落点 + 返回映射 + 两个定位残差。
 *
 * 面板与图层（顺序即绘制顺序，测试按此断言）
 * ------------------------------------------
 * | # | 图层 | 轴 | 内容 |
 * |---|---|---|---|
 * | 1 | line | `scene` | 三维轨道（原始 x/y/z 在显示空间投影；等比例方图） |
 * | 2 | line | `scene` | 截面平面的边框（`plane_corner_*` + `level`，只在本帧给得出平面时） |
 * | 3 | scatter | `scene` | 线性插值点（空心叉） |
 * | 4 | scatter | `scene` | Newton 精化后的截面点（实心） |
 * | 5 | scatter | `scene` | 当前轨道端点 |
 * | 6 | line | `section` | 线性插值点在截面内的位置（空心叉） |
 * | 7 | scatter | `section` | 精化点（实心） |
 * | 8 | line | `retmap` | 返回映射：第 k 次穿越 → 第 k+1 次穿越 |
 * | 9 | scatter | `retmap` | 返回映射点 |
 * | 10 | line | `residual` | 离平面的距离 + 线性插值点→精化点的位移（对数，底部窄条） |
 *
 * 三条刻意的决定
 * --------------
 * 1. **截面上"两个点"必须同时出现**：线性插值点（O(h²)）与 Newton 精化点在截面图上
 *    几乎重合（h=0.01 时位移 0.058 是相空间长度，占图宽百分之几都不到），
 *    **重合本身不是"精化没用"**——所以位移被单独画在底部残差条上，数字写进文本。
 *    拿插值点当精确点是本章的第三个陷阱。
 * 2. **只收单方向穿越**：帧的 `strings.note` 与 `meta.direction` 说明方向敏感；
 *    返回映射若是"双值重影"就说明两个方向的穿越混进来了，所以本模型把
 *    `n_cross`/`n_cross_max` 与"只收由负变正"一起写在图上。
 * 3. **空帧不崩、也不装**：`t=0` 时截面数组长度是 0（还没穿越过），这时相关图层
 *    一个都不画、面板里写明"还没有收到穿越点（n_cross=0）"，而不是画一条 0 残差的线。
 */

import { formatNumber, niceLimits } from "../../shared/src/axes.js";
import { countNonFinite } from "../../shared/src/geometry.js";
import type { Frame } from "../../shared/src/frame.js";
import {
  ModelInputError,
  applyOverflow,
  canvasSize,
  clipBadgeText,
  criticalOf,
  dataSourceOf,
  degradedBadge,
  finiteMinMax,
  finitePositiveMinMax,
  metaNumber,
  palette,
  paramSummary,
  placeBadge,
  requireArray,
  requireDrawableLayers,
  requirePitfalls,
  requireSameLength,
  resolveTextOverlaps,
  sceneShell,
  segmentedSeries,
  toNumeric,
  type Rect,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  criticalBadge,
  lineLayer,
  makeAxis,
  pitfallsBand,
  scatterLayer,
  sourceBadge,
  text,
} from "../../shared/src/scene.js";
import { badgeUnitCap, equalSpanLimits, equalSpanTicks, layout35, projectInto, projectedBoxLimits, projectionOf, stackTexts, type WorldBox } from "./common.js";
import { manifest } from "./manifest.gen.js";

const CHAPTER = "3.5";
const VIZ = "poincare_section";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "Poincaré 截面：线性插值点 vs Newton 精化点";

/**
 * 固定的三维窗口（**与数据无关**，manifest 声明 `autoscaled: false`）。
 * 标定口径：Lorenz（ρ≤45，|x|,|y|≲25、0≤z≲60）与 Rössler（c≤12，|x|,|y|≲17、0≤z≲52）
 * 的轨道都落在 `[-25,25]×[-25,25]×[-5,60]` 内；超出就由 `checkOverflow` 标出来。
 */
const SCENE_BOX: WorldBox = { x: [-25, 25], y: [-25, 25], z: [-5, 60] };
/** ρ 的 Hopf 临界（manifest 里 `rho.critical`）与 c 的临界，帧 `critical` 缺失时回落。 */
const RHO_CRITICAL = 24.736842105263158;
/** 后端给残差数组的地板（`_floor(..., 1e-18)`）：对数轴上的"0"，要画出来并说明。 */
const RESIDUAL_FLOOR = 1e-18;

const SLOT = {
  source: 0,
  sceneAxis: 1,
  sectionAxis: 2,
  retmapAxis: 3,
  critical: 4,
  overflow: 5,
  status: 6,
  degraded: 7,
} as const;

export const poincareSectionModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  // —— 必需数组（截面相关的允许长度为 0：t=0 时还没有穿越）——
  const orbitX = requireArray(frame, "orbit_x");
  const orbitY = requireArray(frame, "orbit_y");
  const orbitZ = requireArray(frame, "orbit_z");
  const secX = requireArray(frame, "section_x");
  const secY = requireArray(frame, "section_y");
  const secZ = requireArray(frame, "section_z");
  const secU = requireArray(frame, "section_inplane_x");
  const secV = requireArray(frame, "section_inplane_y");
  const secT = requireArray(frame, "section_t");
  const resPlane = requireArray(frame, "residual_plane");
  const resShift = requireArray(frame, "residual_shift");
  const linX = requireArray(frame, "lin_x");
  const linY = requireArray(frame, "lin_y");
  const linZ = requireArray(frame, "lin_z");
  const retX = requireArray(frame, "return_x");
  const retY = requireArray(frame, "return_y");
  const pitfalls = requirePitfalls(frame);
  requireSameLength(orbitX, orbitY, "orbit_x 与 orbit_y");
  requireSameLength(orbitX, orbitZ, "orbit_x 与 orbit_z");
  for (const [a, b, what] of [
    [secX, secY, "section_x 与 section_y"],
    [secX, secZ, "section_x 与 section_z"],
    [secX, secU, "section_x 与 section_inplane_x"],
    [secX, secV, "section_x 与 section_inplane_y"],
    [secX, secT, "section_x 与 section_t"],
    [secX, resPlane, "section_x 与 residual_plane"],
    [secX, resShift, "section_x 与 residual_shift"],
    [secX, linX, "section_x 与 lin_x"],
    [secX, linY, "section_x 与 lin_y"],
    [secX, linZ, "section_x 与 lin_z"],
    [retX, retY, "return_x 与 return_y"],
  ] as const) {
    requireSameLength(a, b, what);
  }
  if (retX.length !== 0 && secX.length !== 0 && retX.length !== Math.max(0, secX.length - 1)) {
    throw new ModelInputError(
      `return_x 有 ${retX.length} 个点，但截面点有 ${secX.length} 个（应为 n−1=${Math.max(0, secX.length - 1)}）——返回映射配错了对，拒绝画`,
    );
  }

  // —— 可选字段 ——
  const degraded: string[] = [];
  const proj = projectionOf(frame, opts, degraded);
  const summary = paramSummary(frame, opts, [
    { name: "rho", label: "ρ" },
    { name: "sigma", label: "σ" },
    { name: "c", label: "c" },
  ]);
  degraded.push(...summary.degraded);
  const nCross = metaNumber(frame, "n_cross");
  const nCrossMax = metaNumber(frame, "n_cross_max");
  // 帧用 −1 表示"不适用"（t=0 时一次穿越都没有）：这是哨兵不是读数，不能原样印出去。
  const naNumber = (v: number | undefined): number | undefined => (v === undefined || v < 0 ? undefined : v);
  const meanReturn = naNumber(metaNumber(frame, "mean_return"));
  const maxResidual = naNumber(metaNumber(frame, "max_residual"));
  const maxShift = naNumber(metaNumber(frame, "max_shift"));
  const level = metaNumber(frame, "level");
  const planeIndex = metaNumber(frame, "plane_index");
  const h = metaNumber(frame, "h");
  const tNow = metaNumber(frame, "t_now");
  const hasCrossings = frame.meta["has_crossings"];
  const blowup = frame.meta["blowup"];
  if (nCross === undefined) degraded.push("帧 meta 缺 n_cross：不标收到多少个截面点（不猜）");
  if (h === undefined) degraded.push("帧 meta 缺 h：不标积分步长（不猜）");
  if (level === undefined) degraded.push("帧 meta 缺 level：不标截面位置（不猜）");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: tNow ?? opts.t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  // ------------------------------------------------------------------------- //
  // 排版：左列「三维方图 + 返回映射」，右列截面方图，底部残差窄条
  // ------------------------------------------------------------------------- //
  const panels = layout35(width, height, { columns: true, share: 0.55, columnGap: 58, mainShare: 0.52, gap: 40, strip: true, stripShare: 0.16 });
  const strip = panels.strip as Rect;

  // 三维方图（等比例）
  const sceneSide = Math.max(90, Math.min(panels.main[2], panels.main[3]));
  const sceneRect: Rect = [panels.main[0], panels.main[1], sceneSide, sceneSide];
  const sceneTextX = panels.main[0] + sceneSide + 22;
  const sceneTextW = panels.main[2] - sceneSide - 22;

  const boxLim = projectedBoxLimits(proj, SCENE_BOX, 5);
  scene.axes.push(
    makeAxis("scene", sceneRect, {
      xlim: boxLim.xlim,
      ylim: boxLim.ylim,
      xticks: boxLim.ticks,
      yticks: boxLim.ticks,
      xlabel: "",
      ylabel: "v",
      title: "三维相空间与截面平面",
      autoscaled: false,
    }),
  );

  // —— 图层 1：轨道 ——
  const orb = projectInto(proj, orbitX, orbitY, orbitZ);
  scene.layers.push(lineLayer("scene", segmentedSeries(orb.u, orb.v, pal.reference, { width: 1.1, alpha: 0.9 })));

  // —— 图层 2：截面平面的边框（只有本帧给得出平面时才画）——
  const cornerX = frame.has("plane_corner_x") ? requireArray(frame, "plane_corner_x") : undefined;
  const cornerY = frame.has("plane_corner_y") ? requireArray(frame, "plane_corner_y") : undefined;
  if (cornerX !== undefined && cornerY !== undefined && cornerX.length === 2 && cornerY.length === 2 && level !== undefined && planeIndex === 2) {
    const cx0 = cornerX[0] as number;
    const cx1 = cornerX[1] as number;
    const cy0 = cornerY[0] as number;
    const cy1 = cornerY[1] as number;
    const ring = [projectInto(proj, [cx0, cx1, cx1, cx0, cx0], [cy0, cy0, cy1, cy1, cy0], [level, level, level, level, level])];
    scene.layers.push(lineLayer("scene", segmentedSeries(ring[0].u, ring[0].v, pal.exact, { width: 1.2, dash: [6, 4] })));
  } else if (planeIndex !== undefined && planeIndex !== 2) {
    degraded.push(
      `本帧截面是 ${planeNameOf(planeIndex)}=${fmt(level, 4)}：帧的 plane_corner_x/y 只给 (x,y) 的范围，画不出这个平面的边框——平面上只画落点（不猜）`,
    );
  }

  // —— 图层 3~4：线性插值点与精化点（三维里几乎重合，差值在残差条上）——
  const linP = projectInto(proj, linX, linY, linZ);
  const secP = projectInto(proj, secX, secY, secZ);
  if (linP.u.length > 0) {
    scene.layers.push(scatterLayer("scene", linP.u, linP.v, { r: 3.2, color: pal.muted }));
    scene.layers.push(scatterLayer("scene", secP.u, secP.v, { r: 2.2, color: pal.marker }));
  } else {
    degraded.push("t=0 还没有穿越点：空 ≠ 零残差");
  }
  // —— 图层 5：轨道当前端点 ——
  if (orb.u.length > 0) {
    const last = orb.u.length - 1;
    scene.layers.push(scatterLayer("scene", [orb.u[last] as number], [orb.v[last] as number], { r: 2.6, color: pal.curves[0] }));
  }

  // ------------------------------------------------------------------------- //
  // 右列 `section`：截面内的落点（等比例方图）
  // ------------------------------------------------------------------------- //
  const secSide = Math.max(120, Math.min(panels.side[2], panels.side[3] - 120));
  const secRect: Rect = [panels.side[0], panels.side[1], secSide, secSide];
  const secSpan = equalSpanLimits(secU, secV, 4);
  const inPlane = pickInPlaneAxes(frame);
  scene.axes.push(
    makeAxis("section", secRect, {
      xlim: secSpan.xlim,
      ylim: secSpan.ylim,
      xticks: equalSpanTicks(secSpan.xlim, 4),
      yticks: equalSpanTicks(secSpan.ylim, 4),
      xlabel: inPlane === undefined ? "平面内第一轴" : AXIS_NAME[inPlane[0]],
      ylabel: inPlane === undefined ? "平面内第二轴" : AXIS_NAME[inPlane[1]],
      title: "截面内的落点（原始坐标）",
      autoscaled: true,
    }),
  );
  if (secU.length > 0) {
    if (inPlane !== undefined) {
      const linU = pickArray(frame, LIN_NAMES[inPlane[0]]);
      const linV = pickArray(frame, LIN_NAMES[inPlane[1]]);
      if (linU !== undefined && linV !== undefined) {
        scene.layers.push(
          scatterLayer("section", linU, linV, { r: 3.2, color: pal.muted }),
        );
      } else {
        degraded.push("帧缺 lin_x/lin_y/lin_z：截面图只画精化点（不拿精化点冒充插值点）");
      }
    } else {
      degraded.push("截面内坐标与 section_x/y/z 对不上（帧没给平面内的轴序）：截面图只画精化点");
    }
    scene.layers.push(scatterLayer("section", secU, secV, { r: 2.2, color: pal.marker }));
    const secBlocks: { body: string; color: string; size: number }[] = [
      {
        body:
          `实心=Newton 精化点，空心=线性插值点：位移最大 ${fmt(maxShift, 3)}（相空间长度），本图尺度上重合；` +
          `"看着重合"不等于插值够准，差值画在底部残差条上。`,
        color: pal.warn,
        size: 10,
      },
      {
        body: `只收由负变正的单向穿越（双向会得到双值重影）；到 t=${fmt(tNow, 4)} 共 ${fmt(nCross, 4)} 点。`,
        color: pal.muted,
        size: 10,
      },
    ];
    const st = stackTexts(
      secRect[0],
      secRect[1] + secSide + 30,
      panels.side[2] * 0.98,
      panels.side[1] + panels.side[3] - 2,
      secBlocks,
    );
    scene.texts.push(...st.texts);
    if (st.dropped.length > 0) degraded.push(`截面图下方说明放不下 ${st.dropped.length} 段：未显示`);
  } else {
    scene.texts.push(
      text(secRect[0] + 6, secRect[1] + 4, `到 t=${fmt(tNow, 4)} 为止还没有收到穿越点（n_cross=0）：这一格是空的——不是"零残差"。`, {
        role: "note",
        size: 10,
        color: pal.warn,
        maxWidth: secRect[2] * 0.96,
      }),
    );
  }

  // ------------------------------------------------------------------------- //
  // 左列下半 `retmap`：返回映射
  // ------------------------------------------------------------------------- //
  const retSide = Math.max(80, Math.min(panels.error[2] * 0.42, panels.error[3]));
  const retRect: Rect = [panels.error[0], panels.error[1], retSide, panels.error[3]];
  const retXLim = paddedLimits(retX);
  const retYLim = paddedLimits(retY);
  scene.axes.push(
    makeAxis("retmap", retRect, {
      xlim: retXLim,
      ylim: retYLim,
      xticks: equalSpanTicks(retXLim, 3),
      yticks: equalSpanTicks(retYLim, 3),
      xlabel: "k 次",
      ylabel: "k+1 次",
      title: "返回映射",
      autoscaled: true,
    }),
  );
  if (retX.length > 0) {
    scene.layers.push(
      lineLayer("retmap", segmentedSeries(retX, retY, pal.reference, { width: 0.9, alpha: 0.8 })),
    );
    scene.layers.push(scatterLayer("retmap", retX, retY, { r: 2.2, color: pal.curves[0] }));
  }
  const retTextX = panels.error[0] + retSide + 22;
  const retTextW = panels.error[2] - retSide - 22;
  if (retTextW > 80) {
    const retBlocks: { body: string; color: string; size: number }[] = [
      {
        body:
          `返回映射：横轴 = 第 k 次穿越的平面内第一轴，纵轴 = 第 k+1 次的第二轴（帧口径）；` +
          (meanReturn === undefined
            ? `平均返回时间还不适用（帧里 meta.mean_return = −1 是"没有穿越点"的哨兵，不是读数）。`
            : `平均返回时间 ${fmt(meanReturn, 3)}。`),
        color: pal.ink,
        size: 10,
      },
    ];
    const st = stackTexts(retTextX, panels.error[1] + 2, retTextW, panels.error[1] + panels.error[3] - 2, retBlocks);
    scene.texts.push(...st.texts);
    if (st.dropped.length > 0) degraded.push(`返回映射旁的说明放不下 ${st.dropped.length} 段：未显示`);
  }

  // ------------------------------------------------------------------------- //
  // 底部窄条 `residual`：两个残差（对数）
  // ------------------------------------------------------------------------- //
  const resPositive = finitePositiveMinMax(concat(concat(resPlane, resShift), Float64Array.from([RESIDUAL_FLOOR])));
  const resYlim: [number, number] =
    resPositive === undefined ? [1e-18, 1] : niceLimits(resPositive[0], resPositive[1], 5, { log: true, pad: 0.05 });
  const resXLim: [number, number] = secT.length > 0 ? (finiteMinMax(secT) ?? [0, 1]) : [0, Math.max(1, tNow ?? 1)];
  scene.axes.push(
    makeAxis("residual", strip, {
      xlim: resXLim,
      ylim: resYlim,
      yscale: "log",
      xlabel: "",
      ylabel: "残差",
      title: "",
      autoscaled: true,
    }),
  );
  if (secT.length > 0) {
    scene.layers.push(
      lineLayer("residual", [
        { x: resXLim, y: [RESIDUAL_FLOOR, RESIDUAL_FLOOR], color: pal.muted, width: 1, dash: [1, 3] },
      ]),
    );
    scene.layers.push(
      lineLayer("residual", segmentedSeries(secT, resPlane, pal.marker, { width: 1.2, accept: (v) => v > 0 })),
    );
    scene.layers.push(
      lineLayer("residual", segmentedSeries(secT, resShift, pal.curves[2], { width: 1.2, accept: (v) => v > 0 })),
    );
    const stripLeft = stackTexts(strip[0] + 6, strip[1] + 4, Math.max(160, strip[2] * 0.6), strip[1] + strip[3] - 4, [
      {
        body: `红=离平面距离（max ${fmt(maxResidual, 3)}）　黄=插值→精化位移（max ${fmt(maxShift, 3)}）`,
        color: pal.muted,
        size: 10,
      },
    ]);
    scene.texts.push(...stripLeft.texts);
    const stripRight = stackTexts(strip[0] + Math.max(180, strip[2] * 0.62), strip[1] + 4, Math.max(160, strip[2] * 0.36), strip[1] + strip[3] - 4, [
      {
        body: `RK4 定步长 h=${fmt(h, 3)}；线性插值（O(h²)）+ 3 次 Newton 精化`,
        color: pal.muted,
        size: 10,
      },
    ]);
    scene.texts.push(...stripRight.texts);
  } else {
    scene.texts.push(
      text(strip[0] + 6, strip[1] + 4, `n_cross=0：没有穿越点就没有残差可画（空 ≠ 0）`, { role: "note", size: 10, color: pal.warn, maxWidth: strip[2] }),
    );
  }

  // ------------------------------------------------------------------------- //
  // 三维图旁的说明
  // ------------------------------------------------------------------------- //
  if (sceneTextW > 110) {
    const colBlocks: { body: string; color: string; size: number; role: "label" | "note" }[] = [
      {
        body:
          `${summary.text}　步长 h=${fmt(h, 3)}　截面 ${planeIndex === undefined ? "?" : planeNameOf(planeIndex)}${fmt(level, 4)}` +
          `　穿越只收"由负变正"的单方向点（双向会得到双值重影）：${fmt(nCross, 4)} / 上限 ${fmt(nCrossMax, 4)}`,
        color: pal.ink,
        size: 10,
        role: "label",
      },
      {
        body:
          `正交投影（显示空间，无透视）：方位角 ${proj.azimuthDeg}°、仰角 ${proj.elevationDeg}°（${
            proj.source === "meta" ? "帧 meta" : proj.source === "params" ? "请求值" : "缺省值"
          }）；固定窗口 x,y∈[${SCENE_BOX.x[0]}, ${SCENE_BOX.x[1]}]、z∈[${SCENE_BOX.z[0]}, ${SCENE_BOX.z[1]}]（manifest 声明 autoscaled=false）。` +
          `三维图上插值点与精化点也重合：位移 ${fmt(maxShift, 3)} 在 ${Math.round(sceneSide)}px 的图上不到一个像素。`,
        color: pal.muted,
        size: 10,
        role: "note",
      },
    ];
    const st = stackTexts(sceneTextX, panels.main[1] + 2, sceneTextW, panels.main[1] + panels.main[3] - 2, colBlocks);
    scene.texts.push(...st.texts);
    if (st.dropped.length > 0) degraded.push(`三维图旁的说明列放不下 ${st.dropped.length} 段（列宽 ${Math.round(sceneTextW)}px）：未显示，口径仍见角标`);
  } else {
    degraded.push("画布偏窄：三维图旁的说明列放不下，口径说明只保留角标与面板内文字");
  }

  // ------------------------------------------------------------------------- //
  // 角标（规则 4/5/6）
  // ------------------------------------------------------------------------- //
  scene.badges.push(placeBadge(badge("source", sourceBadge(scene.dataSource).text, 8, 8), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.sceneAxis, width));
  scene.badges.push(placeBadge(badge("autoscale", "截面/返回映射/残差轴：自适应（按数据取整）", 8, 8), SLOT.sectionAxis, width));
  scene.badges.push(
    placeBadge(
      badge(
        "status",
        clipBadgeText(
          `n_cross=${fmt(nCross, 4)}（上限 ${fmt(nCrossMax, 4)}）｜h=${fmt(h, 3)}｜平均返回 ${fmt(meanReturn, 3)}｜最大位移 ${fmt(maxShift, 3)}｜${
            blowup === true ? "检测到发散（blowup）" : hasCrossings === true ? "有穿越" : "尚无穿越"
          }`,
          badgeUnitCap(width),
        ),
        8,
        8,
      ),
      SLOT.retmapAxis,
      width,
    ),
  );
  const rho = metaNumber(frame, "rho");
  if (rho !== undefined) {
    scene.badges.push(placeBadge(criticalBadge("ρ", rho, criticalOf(frame, "rho", RHO_CRITICAL), { beyond: "above" }), SLOT.critical, width));
  } else {
    degraded.push("帧 meta 缺 rho：不标 ρ 的临界值（不猜）");
  }
  const status = degradedBadge(degraded, SLOT.degraded, width);
  if (status !== undefined) scene.badges.push(status);

  requireDrawableLayers(scene);
  applyOverflow(scene, countNonFinite(orbitX) + countNonFinite(secX) + countNonFinite(resShift), { slot: SLOT.overflow, width });

  const panelTop = pitfallsBand(scene)?.[1];
  scene.texts = resolveTextOverlaps(scene.texts, [8, 24, width - 16, (panelTop ?? height) - 28]);
  return scene;
};

// --------------------------------------------------------------------------- //
// 局部工具
// --------------------------------------------------------------------------- //

const AXIS_NAME = ["x", "y", "z"] as const;
const LIN_NAMES = ["lin_x", "lin_y", "lin_z"] as const;

function pickArray(frame: Frame, name: string): ArrayLike<number> | undefined {
  return frame.has(name) ? toNumeric(frame.array(name)) : undefined;
}

/** 平面名（`plane_index` 0/1/2 → x=const / y=const / z=const）。 */
function planeNameOf(planeIndex: number): string {
  const i = Math.round(planeIndex);
  return i >= 0 && i < 3 ? `${AXIS_NAME[i]} = ` : "平面 ";
}

/**
 * 截面内坐标是哪两个原始坐标轴（**从数据里认**，不写死平面方向）。
 *
 * 帧把 `section_inplane_x/y` 定义为"平面内第一/第二轴"，但不同系统的平面不同
 * （Lorenz 用 z=const → 平面内是 (x,y)；Rössler 用 x=0 → 平面内是 (y,z)）。
 * 这里逐个比对 `section_x/y/z` 与 `section_inplane_*`（同一批点、同一顺序），
 * 认出来之后才能把 `lin_*` 摆到同一对轴上——写死 (x,y) 会在 Rössler 上画错。
 */
function pickInPlaneAxes(frame: Frame): [number, number] | undefined {
  const inU = pickArray(frame, "section_inplane_x");
  const inV = pickArray(frame, "section_inplane_y");
  if (inU === undefined || inV === undefined || inU.length === 0) return undefined;
  const coords: (ArrayLike<number> | undefined)[] = [
    pickArray(frame, "section_x"),
    pickArray(frame, "section_y"),
    pickArray(frame, "section_z"),
  ];
  const find = (target: ArrayLike<number>): number | undefined => {
    for (let k = 0; k < coords.length; k += 1) {
      const c = coords[k];
      if (c === undefined || c.length !== target.length) continue;
      if (sameValues(c, target)) return k;
    }
    return undefined;
  };
  const ix = find(inU);
  const iy = find(inV);
  if (ix === undefined || iy === undefined || ix === iy) return undefined;
  return [ix, iy];
}

function sameValues(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
  let scale = 0;
  for (let i = 0; i < b.length; i += 1) scale = Math.max(scale, Math.abs(b[i] as number));
  const tol = 1e-9 * Math.max(1, scale);
  for (let i = 0; i < a.length; i += 1) {
    const d = Math.abs((a[i] as number) - (b[i] as number));
    if (!(d <= tol)) return false;
  }
  return true;
}

/** 数据驱动的 1-2-5 取整范围（空数组时给退化范围，避免 0 宽轴）。 */
function paddedLimits(a: ArrayLike<number>): [number, number] {
  const mm = finiteMinMax(a);
  if (mm === undefined) return [-1, 1];
  const lo = mm[0];
  const hi = mm[1];
  if (!(hi > lo)) {
    const d = Math.max(Math.abs(lo), 1) * 0.1;
    return niceLimits(lo - d, lo + d, 4);
  }
  return niceLimits(lo, hi, 4, { pad: 0.08 });
}

function fmt(v: number | undefined, sig: number): string {
  return v === undefined ? "?" : formatNumber(v, sig);
}

function concat(a: ArrayLike<number>, b: ArrayLike<number>): Float64Array {
  const out = new Float64Array(a.length + b.length);
  for (let i = 0; i < a.length; i += 1) out[i] = a[i] as number;
  for (let i = 0; i < b.length; i += 1) out[a.length + i] = b[i] as number;
  return out;
}
