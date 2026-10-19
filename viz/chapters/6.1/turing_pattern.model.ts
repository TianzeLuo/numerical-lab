/**
 * `turing_pattern`（6.1 章 viz 3）：Gray--Scott 斑图 + 色散关系 + 模板误差。
 *
 * 三条纪律
 * -------
 * 1. **“有斑图”不是图灵判据**：真图灵要同时满足底态无扩散稳定与正波数 λ*>0。
 *    `stripes` 的底态存在但 λ(0)=+0.0054（均匀失稳），`spots` 没有非平凡稳态，
 *    `uniform` 的 λ 恒为 0；三种都画在同一个页面上作对照。
 * 2. **波长无定义时不写 0**：`wavelength_valid=0` 时帧里存的是占位 0，模型不画
 *    “波长=0”，而是写“无空间结构/无定义”。
 * 3. **离散误差是空间模板的**：右下的五点模板相对误差随 n 按 O(n^-2) 收敛；
 *    它不是完整时间推进误差。dt 被夹紧时必须显示请求值与实际上限。
 *
 * 帧 `time.mode="scrub"`：`t` 是显式步数，不是物理时间。
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
  heatLayer,
  lineLayer,
  makeAxis,
  scatterLayer,
  sourceBadge,
} from "../../shared/src/scene.js";
import { axisTitleOf, axisYlabelOf, columnsLayout, textBlock, unitBudget } from "../_common/panel.js";
import {
  CHAPTER,
  HEAT_RIGHT_GUTTER,
  SLOT,
  extentOf,
  fmt,
  heatColumn,
  padRange,
  pushGroups,
  singleColumn,
  statusLine,
} from "./common.js";
import { manifest } from "./manifest.gen.js";

const VIZ = "turing_pattern";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "Gray--Scott：图灵斑图、色散关系与网格误差";

export const turingPatternModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  const x = requireArray(frame, "x");
  const y = requireArray(frame, "y");
  const v = requireArray(frame, "v");
  const phi = requireArray(frame, "phi");
  const rate = requireArray(frame, "rate");
  const nError = requireArray(frame, "n_error");
  const error = requireArray(frame, "error");
  requireSameLength(phi, rate, "phi 与 rate");
  requireSameLength(nError, error, "n_error 与 error");
  const n = Math.max(1, Math.round(Math.sqrt(v.length)));
  if (n * n !== v.length) throw new Error(`v 长度 ${v.length} 不是 n²`);
  if (x.length !== n || y.length !== n) throw new Error(`x/y 长度 ${x.length}/${y.length} ≠ n=${n}`);
  if (phi.length < 8) throw new Error(`色散曲线采样太少：${phi.length}`);
  const pitfalls = requirePitfalls(frame);

  const degraded: string[] = [];
  const preset = (frame.strings["preset"] as string | undefined) ?? "?";
  const steps = metaNumber(frame, "t") ?? opts.t;
  const stepsInt = metaNumber(frame, "steps") ?? steps;
  const physical = metaNumber(frame, "time_physical");
  const dt = metaNumber(frame, "dt");
  const dtReq = metaNumber(frame, "dt_requested");
  const dtLimit = metaNumber(frame, "dt_limit");
  const dtClamped = (metaNumber(frame, "dt_clamped") ?? 0) > 0.5;
  const phiStar = metaNumber(frame, "phi_star");
  const lambdaStar = metaNumber(frame, "lambda_star");
  const uniformRate = metaNumber(frame, "uniform_rate");
  const baseValid = (metaNumber(frame, "base_valid") ?? 0) > 0.5;
  const baseU = metaNumber(frame, "base_u");
  const baseV = metaNumber(frame, "base_v");
  const turingBand = (metaNumber(frame, "turing_band") ?? 0) > 0.5;
  const amplitude = metaNumber(frame, "amplitude");
  const spots = metaNumber(frame, "spot_count");
  const wavelength = metaNumber(frame, "wavelength");
  const wavelengthValid = (metaNumber(frame, "wavelength_valid") ?? 0) > 0.5;
  const vmaxRaw = metaNumber(frame, "vmax") ?? 1;
  const effectivelyZero = (amplitude ?? 0) < 1e-12 && vmaxRaw < 1e-6;
  const vmax = effectivelyZero ? 1 : Math.max(1e-9, vmaxRaw);
  const slope = metaNumber(frame, "grid_error_slope");
  const status = (frame.strings["status"] as string | undefined) ?? "?";
  const uniformLike = !baseValid && Math.abs(uniformRate ?? 0) < 1e-12;
  const regimeLine = turingBand
    ? "真图灵带：底态稳，φ*>0。"
    : uniformLike
      ? "平凡态：反应项为零，纯扩散衰减。"
      : baseValid && (uniformRate ?? 0) > 0
        ? "均匀模失稳，非图灵。"
        : baseValid
          ? "底态稳定，但无正波数增长。"
          : "无非平凡稳态：自复制瞬态。";
  const finite = (metaNumber(frame, "finite") ?? 1) > 0.5;
  if (phiStar === undefined) degraded.push("帧缺 φ*：不画最不稳定波数标记");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: steps,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const wide = width >= 820;
  const L = wide ? columnsLayout(scene, width, height, { topShare: 0.62, auxH: 160, gap: 56 }) : null;
  const S = wide ? null : singleColumn(scene, width, height, 0.58);
  const topY = L ? L.topY : (S as NonNullable<typeof S>).topY;
  const floorY = L ? L.floor : (S as NonNullable<typeof S>).floor;
  const narrowHeat = L
    ? null
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
  const mainRect: [number, number, number, number] = L
    ? heatColumn(L.left, L.leftW, L.topY, L.topH, 150).field
    : (narrowHeat as NonNullable<typeof narrowHeat>).field;
  const inner = L ? heatColumn(L.left, L.leftW, L.topY, L.topH, 150) : narrowHeat;
  const errRect: [number, number, number, number] = L ? [L.left, L.botY, L.leftW, L.botH] : (S as NonNullable<typeof S>).aux;

  const cell = 1 / Math.max(1, metaNumber(frame, "n") ?? n);
  const xLo = (x[0] as number) - cell / 2;
  const xHi = (x[x.length - 1] as number) + cell / 2;
  const yLo = (y[0] as number) - cell / 2;
  const yHi = (y[y.length - 1] as number) + cell / 2;
  const main = makeAxis("main", mainRect, {
    xlim: [xLo, xHi],
    ylim: [yLo, yHi],
    xlabel: "x",
    ylabel: axisYlabelOf(SPEC?.axes, "main", "y"),
    title: axisTitleOf(SPEC?.axes, "main", "v(x,y)：Gray--Scott 斑图"),
    autoscaled: false,
    tickTarget: 4,
  });
  const errRange = extentOf([error], [1e-3, 1]);
  const errAxis = makeAxis("error", errRect, {
    xlim: [Math.min(nError[0] as number, 64), Math.max(nError[nError.length - 1] as number, 128)],
    ylim: [Math.max(1e-10, errRange[0] * 0.4), Math.max(errRange[0] * 40, errRange[1] * 2.5)],
    yscale: "log",
    xlabel: "网格 n",
    ylabel: axisYlabelOf(SPEC?.axes, "error", "相对误差"),
    title: axisTitleOf(SPEC?.axes, "error", "五点模板增长率误差 vs n（对数纵轴）"),
    autoscaled: true,
    tickTarget: 4,
  });
  scene.axes.push(main, errAxis);

  scene.layers.push(heatLayer("main", v, n, n, [xLo, xHi, yLo, yHi], { cmap: "viridis", vmin: 0, vmax }));

  scene.layers.push(
    lineLayer("error", { x: nError, y: error, color: pal.residual, width: 1.8 }),
    scatterLayer("error", nError, error, { r: 3, color: pal.residual }),
  );
  if (slope !== undefined && Number.isFinite(slope) && error.length >= 2) {
    const n0 = nError[nError.length - 1] as number;
    const e0 = error[error.length - 1] as number;
    if (n0 > 0 && e0 > 0) {
      const refX = new Float64Array(nError.length);
      const refY = new Float64Array(nError.length);
      for (let i = 0; i < nError.length; i += 1) {
        const ni = nError[i] as number;
        refX[i] = ni;
        refY[i] = e0 * Math.pow(ni / n0, -2);
      }
      scene.layers.push(lineLayer("error", { x: refX, y: refY, color: pal.reference, width: 1.1, dash: [5, 4] }));
    }
  }

  let dispAxis: ReturnType<typeof makeAxis> | undefined;
  if (L) {
    const dispH = Math.max(100, Math.min(150, Math.round((floorY - topY) * 0.38)));
    const rateRange = extentOf([rate], [-1, 1]);
    const lo = Math.min(0, rateRange[0]);
    const hi = Math.max(0, rateRange[1]);
    dispAxis = makeAxis("dispersion", [L.rightX, topY, L.rightW, dispH], {
      xlim: [0, Math.max(1, phi[phi.length - 1] as number)],
      ylim: padRange(lo, hi, 0.08),
      xlabel: "波数 φ",
      ylabel: axisYlabelOf(SPEC?.axes, "dispersion", "λ_max"),
      title: axisTitleOf(SPEC?.axes, "dispersion", "解析色散关系 λ(φ)"),
      autoscaled: true,
      tickTarget: 4,
    });
    scene.axes.push(dispAxis);
    scene.layers.push(lineLayer("dispersion", { x: phi, y: rate, color: pal.curves[0] as string, width: 1.7 }));
    scene.layers.push(
      lineLayer("dispersion", { x: [0, phi[phi.length - 1] as number], y: [0, 0], color: pal.reference, width: 0.9, dash: [3, 3] }),
    );
    if (phiStar !== undefined && phiStar > 0.5 && phiStar <= (dispAxis.xlim[1] as number)) {
      scene.layers.push(
        lineLayer("dispersion", { x: [phiStar, phiStar], y: [dispAxis.ylim[0], dispAxis.ylim[1]], color: pal.exact, width: 1.2, dash: [4, 3] }),
      );
      if (lambdaStar !== undefined && lambdaStar >= (dispAxis.ylim[0] as number) && lambdaStar <= (dispAxis.ylim[1] as number)) {
        scene.layers.push(scatterLayer("dispersion", [phiStar], [lambdaStar], { r: 3.4, color: pal.marker }));
      }
    }
  } else {
    degraded.push("窄画布：省略色散关系与右栏说明");
  }

  if (inner !== null && inner.innerW >= 120) {
    const budget = unitBudget(inner.innerW);
    let yy = topY;
    const push = (lines: readonly string[], color: string): void => {
      const block = textBlock(inner.innerX, yy, inner.innerBottom, lines, color, budget, "top");
      scene.texts.push(...block);
      yy += block.length * 15;
    };
    if (!L) {
      const shortRegime = turingBand
        ? "真图灵带。"
        : uniformLike
          ? "平凡态：纯扩散。"
          : baseValid && (uniformRate ?? 0) > 0
            ? "均匀模失稳。"
            : baseValid
              ? "底态稳；无带。"
              : "自复制瞬态。";
      push(
        [
          effectivelyZero ? "色=v≡0（色标固定 0…1）。" : `色=v(x,y)，色标 0…${fmt(vmax, 3)}。`,
          `φ*=${fmt(phiStar, 2)}，λ*=${fmt(lambdaStar, 4)}。`,
          `${shortRegime} t=${fmt(physical, 2)}（${fmt(stepsInt, 0)} 步）。`,
          wavelengthValid ? `振幅 ${fmt(amplitude, 3)}；波长 ${fmt(wavelength, 3)}。` : `振幅 ${fmt(amplitude, 3)}；主波长无定义。`,
        ],
        pal.ink,
      );
    } else {
      push(
        [
          effectivelyZero ? "色=v≡0（色标固定 0…1）。" : `色=v(x,y)，色标 0…${fmt(vmax, 3)}。`,
          `φ*=${fmt(phiStar, 2)}，λ*=${fmt(lambdaStar, 4)}。`,
          `λ(0)=${fmt(uniformRate, 4)}。`,
          baseValid ? `底态 (${fmt(baseU, 3)}, ${fmt(baseV, 3)})。` : uniformLike ? "平凡态 (1,0)。" : "无稳态。",
          regimeLine,
          `t=${fmt(physical, 2)}（${fmt(stepsInt, 0)} 步）。`,
          `振幅 ${fmt(amplitude, 3)}；${fmt(spots, 0)} 个极大。`,
        ],
        pal.ink,
      );
      push(
        wavelengthValid
          ? [
              phiStar !== undefined && phiStar > 0.5
                ? `波长 ${fmt(wavelength, 3)}；理论 ${fmt((2 * Math.PI) / phiStar, 3)}。`
                : `波长 ${fmt(wavelength, 3)}（无图灵理论）。`,
            ]
          : ["主波长无定义（不是 0）。"],
        pal.muted,
      );
    }
  }

  if (L && L.rightW >= 200) {
    pushGroups(
      scene,
      L.rightX,
      (dispAxis as NonNullable<typeof dispAxis>).rect[1] + (dispAxis as NonNullable<typeof dispAxis>).rect[3] + 36,
      floorY,
      L.rightW,
      [
        {
          lines: [
            `φ*=${fmt(phiStar, 2)}，λ*=${fmt(lambdaStar, 4)}。`,
            `λ(0)=${fmt(uniformRate, 4)}。`,
            regimeLine,
          ],
          color: pal.ink,
        },
        {
          lines: [
            dtClamped
              ? `dt ${fmt(dtReq, 2)}→${fmt(dt, 5)}（上限 ${fmt(dtLimit, 5)}）。`
              : `dt=${fmt(dt, 5)}（上限 ${fmt(dtLimit, 5)}）。`,
            `振幅 ${fmt(amplitude, 3)}；${fmt(spots, 0)} 个极大。`,
            wavelengthValid ? `波长 ${fmt(wavelength, 3)}。` : "主波长无定义。",
          ],
          color: pal.muted,
        },
        {
          lines: [
            `五点模板误差阶 ${fmt(slope, 2)}。`,
            "蓝=λ(φ)；灰虚=0；橙虚=φ*。",
            `空间模板误差，非时间推进。状态 ${status}。`,
          ],
          color: finite ? pal.muted : pal.warn,
        },
      ],
      pal,
    );
  } else if (L) {
    degraded.push("右侧说明列过窄：读数并入角标");
  }

  scene.badges.push(placeBadge(sourceBadge(dataSourceOf(frame)), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge(
        "status",
        statusLine([
          preset,
          `φ*=${fmt(phiStar, 2)}`,
          `λ*=${fmt(lambdaStar, 4)}`,
          turingBand ? "图灵带" : "非图灵",
          `步数 ${fmt(stepsInt, 0)}`,
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
