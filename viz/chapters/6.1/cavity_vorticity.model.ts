/**
 * `cavity_vorticity`（6.1 章 viz 1）：顶盖驱动方腔的稳态涡量场与耦合残差。
 *
 * 三条纪律
 * -------
 * 1. **残差不是误差**：左下图是 SOR 的相对残差（已经是 log10），侧栏同时给出与
 *    Ghia 中心线的 RMSE。残差 1e-7 与 RMSE 0.227 同时成立，页面不挑一个好看的数。
 * 2. **热图、等值线、箭头各说一件事**：色=涡量（对称发散色标），灰线=流函数等值线
 *    （只给形状），黑箭头=速度方向（长度按全场最大速度归一，不表示量级）。
 * 3. **时间步约束有两个来源**：扩散 dt~Re h² 与 CFL dt~h/|u| 的交叉点在
 *    Re_x = 2(n−1)；Re=1000/n=41 已在 CFL 一侧，侧栏写明当前由哪条绑定。
 *
 * 帧 `time.mode="none"`；所有物理量都来自后端，模型只做显示几何。
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
  contourLayer,
  heatLayer,
  lineLayer,
  makeAxis,
  scatterLayer,
  sourceBadge,
  vectorFieldLayer,
} from "../../shared/src/scene.js";
import {
  axisTitleOf,
  axisYlabelOf,
  columnsLayout,
  textBlock,
  unitBudget,
} from "../_common/panel.js";
import {
  CHAPTER,
  HEAT_RIGHT_GUTTER,
  SLOT,
  extentOf,
  fmt,
  heatColumn,
  pushGroups,
  singleColumn,
  statusLine,
  symmetricExtent,
} from "./common.js";
import { manifest } from "./manifest.gen.js";

const VIZ = "cavity_vorticity";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "顶盖驱动方腔：涡量、流线与残差";

export const cavityVorticityModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  const x = requireArray(frame, "x");
  const y = requireArray(frame, "y");
  const omega = requireArray(frame, "vorticity");
  const psi = requireArray(frame, "psi");
  const arrowX = requireArray(frame, "arrow_x");
  const arrowY = requireArray(frame, "arrow_y");
  const arrowU = requireArray(frame, "arrow_u");
  const arrowV = requireArray(frame, "arrow_v");
  const residual = requireArray(frame, "residual");
  const residualIndex = requireArray(frame, "residual_index");
  requireSameLength(residual, residualIndex, "residual 与 residual_index");
  if (psi.length !== omega.length) {
    throw new Error(`psi 长度 ${psi.length} ≠ vorticity 长度 ${omega.length}`);
  }
  const n = Math.max(1, Math.round(Math.sqrt(omega.length)));
  if (n * n !== omega.length) throw new Error(`vorticity 长度 ${omega.length} 不是 n²`);
  if (psi.length !== n * n) throw new Error(`psi 长度 ${psi.length} ≠ ${n * n}`);
  if (x.length !== n || y.length !== n) throw new Error(`x/y 长度 ${x.length}/${y.length} ≠ n=${n}`);
  const na = Math.max(1, Math.round(Math.sqrt(arrowU.length)));
  if (na * na !== arrowU.length || arrowV.length !== arrowU.length) {
    throw new Error(`箭头场长度 ${arrowU.length}/${arrowV.length} 不是同一个方阵`);
  }
  if (arrowX.length !== na || arrowY.length !== na) throw new Error(`箭头坐标长度与 ${na}² 不符`);
  const pitfalls = requirePitfalls(frame);

  const degraded: string[] = [];
  const re = metaNumber(frame, "re");
  const nMeta = metaNumber(frame, "n") ?? n;
  const h = metaNumber(frame, "h");
  const sweeps = metaNumber(frame, "sweeps");
  const poisson = metaNumber(frame, "poisson");
  const residualFinal = metaNumber(frame, "residual_final");
  const residualTol = metaNumber(frame, "residual_tol");
  const rmse = metaNumber(frame, "rmse");
  const rmseU = metaNumber(frame, "rmse_u");
  const rmseV = metaNumber(frame, "rmse_v");
  const centerX = metaNumber(frame, "center_x");
  const centerY = metaNumber(frame, "center_y");
  const centerValid = (metaNumber(frame, "center_valid") ?? 0) > 0.5;
  const omegaMax = metaNumber(frame, "omega_abs_max");
  const psiMin = metaNumber(frame, "psi_min");
  const psiMax = metaNumber(frame, "psi_max");
  const dtCfl = metaNumber(frame, "dt_cfl");
  const dtDiff = metaNumber(frame, "dt_diffusion");
  const reCross = metaNumber(frame, "re_cross");
  const binding = metaNumber(frame, "dt_binding") ?? 0;
  const cellRe = metaNumber(frame, "cell_re");
  const centralThreshold = metaNumber(frame, "central_threshold_re");
  const converged = (metaNumber(frame, "converged") ?? 0) > 0.5;
  const status = (frame.strings["status"] as string | undefined) ?? "?";
  if (centerX === undefined || centerY === undefined) degraded.push("帧缺主涡坐标：不画涡心标记");
  if (rmse === undefined) degraded.push("帧缺 RMSE：侧栏不报离散误差");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: 0,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const wide = width >= 820;
  const L = wide ? columnsLayout(scene, width, height, { topShare: 0.62, auxH: 120, gap: 56 }) : null;
  const S = wide ? null : singleColumn(scene, width, height, 0.56);
  const heat = L
    ? heatColumn(L.left, L.leftW, L.topY, L.topH, 150)
    : (() => {
        const Sn = S as NonNullable<typeof S>;
        const fieldW = Math.max(120, Math.min(Sn.main[2] - 110, Math.round(Sn.main[3] * 1.8)));
        const innerX = Sn.main[0] + fieldW + HEAT_RIGHT_GUTTER + 8;
        return {
          field: [Sn.main[0], Sn.main[1], fieldW, Sn.main[3]] as [number, number, number, number],
          innerX,
          innerW: Math.max(0, width - 24 - innerX),
          innerBottom: Sn.main[1] + Sn.main[3],
        };
      })();
  const auxRect: [number, number, number, number] = L ? [L.left, L.botY, L.leftW, L.botH] : (S as NonNullable<typeof S>).aux;

  const main = makeAxis("main", heat.field, {
    xlim: [x[0] as number, x[x.length - 1] as number],
    ylim: [y[0] as number, y[y.length - 1] as number],
    xlabel: "x",
    ylabel: axisYlabelOf(SPEC?.axes, "main", "y"),
    title: axisTitleOf(SPEC?.axes, "main", "稳态涡量与流函数等值线"),
    autoscaled: false,
    tickTarget: 4,
  });
  const aux = makeAxis("residual", auxRect, {
    xlim: [0, Math.max(1, residualIndex[residualIndex.length - 1] as number)],
    ylim: [Math.min(-8, (metaNumber(frame, "residual_final") !== undefined ? Math.log10(Math.max(residualFinal ?? 1e-14, 1e-14)) : -8) - 1), 0],
    xlabel: "耦合迭代（记录点，已降采样）",
    ylabel: axisYlabelOf(SPEC?.axes, "residual", "log10 相对残差"),
    title: "耦合迭代残差（log10；橙虚=容差）",
    autoscaled: true,
  });
  scene.axes.push(main, aux);

  const [omegaLo, omegaHi] = symmetricExtent([omega]);
  scene.layers.push(
    heatLayer("main", omega, n, n, [x[0] as number, x[x.length - 1] as number, y[0] as number, y[y.length - 1] as number], {
      cmap: "rdbu",
      vmin: omegaLo,
      vmax: omegaHi,
    }),
  );

  const [pLo, pHi] = extentOf([psi], [-1, -1]);
  if (pHi > pLo) {
    const levels: number[] = [];
    for (let i = 1; i <= 15; i += 1) levels.push(pLo + ((pHi - pLo) * i) / 16);
    scene.layers.push(
      contourLayer("main", psi, n, n, [x[0] as number, x[x.length - 1] as number, y[0] as number, y[y.length - 1] as number], {
        levels,
        color: pal.ink,
        width: 0.7,
        label: "流函数 ψ 的等值线",
      }),
    );
  } else {
    degraded.push("流函数场退化为常数：不画等值线");
  }

  // 箭头：方向来自后端，长度按全场最大速度归一——只表方向，不表量级。
  let maxSpeed = 0;
  for (let i = 0; i < arrowU.length; i += 1) {
    const uu = arrowU[i] as number;
    const vv = arrowV[i] as number;
    if (!Number.isFinite(uu) || !Number.isFinite(vv)) continue;
    maxSpeed = Math.max(maxSpeed, Math.hypot(uu, vv));
  }
  if (maxSpeed > 0) {
    const display = new Float64Array(arrowU.length);
    const displayV = new Float64Array(arrowV.length);
    const target = 0.045;
    for (let i = 0; i < arrowU.length; i += 1) {
      display[i] = ((arrowU[i] as number) / maxSpeed) * target;
      displayV[i] = ((arrowV[i] as number) / maxSpeed) * target;
    }
    scene.layers.push(
      vectorFieldLayer(
        "main",
        display,
        displayV,
        na,
        na,
        [arrowX[0] as number, arrowX[arrowX.length - 1] as number, arrowY[0] as number, arrowY[arrowY.length - 1] as number],
        { color: pal.ink, stride: 1, scale: 1 },
      ),
    );
  } else {
    degraded.push("速度场全零：不画方向箭头");
  }

  if (centerValid && centerX !== undefined && centerY !== undefined) {
    scene.layers.push(scatterLayer("main", [centerX], [centerY], { r: 3.4, color: pal.marker, label: "主涡中心" }));
  }

  scene.layers.push(
    lineLayer("residual", { x: residualIndex, y: residual, color: pal.curves[0] as string, width: 1.8, label: "相对残差" }),
  );
  if (residual.length > 0) {
    scene.layers.push(
      scatterLayer("residual", [residualIndex[residual.length - 1] as number], [residual[residual.length - 1] as number], {
        r: 3.2,
        color: pal.marker,
        label: "当前残差",
      }),
    );
  }
  if (residualTol !== undefined && residualTol > 0) {
    const tolLog = Math.log10(residualTol);
    if (tolLog >= aux.ylim[0] && tolLog <= aux.ylim[1]) {
      scene.layers.push(lineLayer("residual", { x: [aux.xlim[0], aux.xlim[1]], y: [tolLog, tolLog], color: pal.exact, width: 1.1, dash: [5, 4], label: "收敛容差" }));
    }
  }

  if (heat.innerW >= 120) {
    const budget = unitBudget(heat.innerW);
    let yy = heat.field[1];
    const push = (lines: readonly string[], color: string): void => {
      const block = textBlock(heat.innerX, yy, heat.innerBottom, lines, color, budget, "top");
      scene.texts.push(...block);
      yy += block.length * 15;
    };
    push(
      [
        "色=ω（红正蓝负），灰线=ψ。",
        `ψ∈[${fmt(psiMin, 3)}, ${fmt(psiMax, 3)}]。`,
        "ψ 等值线 15 级，只给形状。",
        "黑箭头=速度方向（全场归一）。",
        centerValid ? `主涡 (${fmt(centerX, 4)}, ${fmt(centerY, 4)})。` : "主涡未检出。",
        `色标 ±${fmt(omegaMax, 3)}。`,
      ],
      pal.ink,
    );
  } else if (L) {
    degraded.push("热图右侧内栏过窄：流场图例并入右栏");
  }

  if (L && L.rightW >= 200) {
    pushGroups(
      scene,
      L.rightX,
      L.topY,
      L.floor,
      L.rightW,
      [
        {
          lines: [
            `Re=${fmt(re, 0)}，n=${fmt(nMeta, 0)}，h=${fmt(h, 4)}。`,
            `SOR ${fmt(sweeps, 0)}；DST Poisson ${fmt(poisson, 0)} 次。`,
            `终态残差 ${fmt(residualFinal, 2)}（tol ${fmt(residualTol, 1)}）。`,
          ],
          color: pal.ink,
        },
        {
          lines: [
            `中心线 RMSE ${fmt(rmse, 4)}。`,
            `u ${fmt(rmseU, 4)}；v ${fmt(rmseV, 4)}。`,
            "残差小 ≠ 解准。",
            "残差量方程，RMSE 量解。",
          ],
          color: pal.muted,
        },
        {
          lines: [
            `dt 扩散=${fmt(dtDiff, 4)}；CFL=${fmt(dtCfl, 4)}。`,
            `交叉 Re_x=${fmt(reCross, 0)}；当前由${binding > 0.5 ? " CFL" : "扩散"}限制。`,
          ],
          color: pal.muted,
        },
        {
          lines: [
            `Re_h=${fmt(cellRe, 2)}；中心格式阈值 Re=${fmt(centralThreshold, 0)}。`,
            cellRe !== undefined && cellRe > 2 ? "本网格不能用中心差分。" : "本网格仍在中心格式稳定侧。",
          ],
          color: pal.muted,
        },
        {
          lines: [
            "蓝=log10 残差；红=末次。",
            `橙虚=容差；状态 ${status}。`,
          ],
          color: pal.muted,
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
          `Re=${fmt(re, 0)}`,
          `n=${fmt(nMeta, 0)}`,
          `残差 ${fmt(residualFinal, 1)}`,
          `RMSE ${fmt(rmse, 3)}`,
          converged ? "已收敛" : "未收敛",
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
