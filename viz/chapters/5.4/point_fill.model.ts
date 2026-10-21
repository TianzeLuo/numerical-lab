/**
 * `point_fill`（5.4 章 viz 1）：MC / Sobol / 秩一格点铺点与误差- N 曲线。
 *
 * 三条纪律
 * -------
 * 1. **“点更均匀”不是结论**：均匀性看偏差 D*_N（Koksma-Hlawka），收敛率看误差-N 斜率；
 *    两件事在图上分开（左点集、右误差曲线）。
 * 2. **误差棒只在 randomize=True 时存在**：关掉随机化，R 个复制逐位相同，标准差恒为 0；
 *    帧里 band_valid=0，占位值不是“误差很小”。侧栏如实说。
 * 3. **参考线只对 MC 是理论**：sigma/sqrt(N) 对 QMC 只是标度参照；QMC 的斜率可以更陡，
 *    但在有限 N 上并非无限好。
 *
 * 帧 `time.mode="scrub"`：`t` 是已铺开的点数 k。
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
  bandLayer,
  lineLayer,
  makeAxis,
  scatterLayer,
  sourceBadge,
} from "../../shared/src/scene.js";
import { TEXT_ROW_H, axisTitleOf, axisYlabelOf, columnsLayout, textBlock, unitBudget } from "../_common/panel.js";
import { CHAPTER, SLOT } from "./common.js";
import { manifest } from "./manifest.gen.js";

const VIZ = "point_fill";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "点集填充对决";
const METHOD_LABEL: Record<string, string> = { mc: "MC", sobol: "Sobol'", lattice: "秩一格点", grid: "张量格点" };

export const pointFillModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  const mcX = requireArray(frame, "mc_x");
  const mcY = requireArray(frame, "mc_y");
  const sbX = requireArray(frame, "sobol_x");
  const sbY = requireArray(frame, "sobol_y");
  const laX = requireArray(frame, "lattice_x");
  const laY = requireArray(frame, "lattice_y");
  requireSameLength(mcX, mcY, "mc_x 与 mc_y");
  requireSameLength(sbX, sbY, "sobol_x 与 sobol_y");
  requireSameLength(laX, laY, "lattice_x 与 lattice_y");
  const nAxis = requireArray(frame, "n_axis");
  const errSel = requireArray(frame, "error");
  const errValid = requireArray(frame, "error_valid");
  const errMc = requireArray(frame, "error_mc");
  const errSb = requireArray(frame, "error_sobol");
  const errLa = requireArray(frame, "error_lattice");
  const errGr = requireArray(frame, "error_grid");
  const errRef = requireArray(frame, "error_ref");
  requireSameLength(nAxis, errSel, "n_axis 与 error");
  requireSameLength(nAxis, errValid, "n_axis 与 error_valid");
  for (const [name, arr] of [
    ["error_mc", errMc],
    ["error_sobol", errSb],
    ["error_lattice", errLa],
    ["error_grid", errGr],
    ["error_ref", errRef],
  ] as const) {
    requireSameLength(nAxis, arr, `n_axis 与 ${name}`);
  }
  const bandLo = frame.has("band_lo") ? requireArray(frame, "band_lo") : undefined;
  const bandHi = frame.has("band_hi") ? requireArray(frame, "band_hi") : undefined;
  const bandValid = frame.has("band_valid") ? requireArray(frame, "band_valid") : undefined;
  if (bandLo !== undefined && bandHi !== undefined && bandValid !== undefined) {
    requireSameLength(nAxis, bandLo, "n_axis 与 band_lo");
    requireSameLength(nAxis, bandHi, "n_axis 与 band_hi");
    requireSameLength(nAxis, bandValid, "n_axis 与 band_valid");
  }
  const pitfalls = requirePitfalls(frame);

  const degraded: string[] = [];
  if (bandLo === undefined) degraded.push("帧缺 band_lo/hi：不画误差带");
  const t = metaNumber(frame, "t") ?? opts.t;
  const nShown = metaNumber(frame, "n_shown") ?? mcX.length;
  const nMax = metaNumber(frame, "n_max") ?? 2048;
  const method = stringOf(frame, "method") ?? "sobol";
  const methodName = stringOf(frame, "method_name") ?? METHOD_LABEL[method] ?? method;
  const integrand = stringOf(frame, "integrand_label") ?? stringOf(frame, "integrand") ?? "?";
  const colon = integrand.indexOf("：");
  const integrandName = colon >= 0 ? integrand.slice(0, colon) : integrand;
  const integrandExpr = colon >= 0 ? integrand.slice(colon + 1).trim() : "";
  const slopes: Record<string, number | undefined> = {
    mc: metaNumber(frame, "slope_mc"),
    sobol: metaNumber(frame, "slope_sobol"),
    lattice: metaNumber(frame, "slope_lattice"),
    grid: metaNumber(frame, "slope_grid"),
  };
  const slopeRef = metaNumber(frame, "slope_reference");
  const slopeSel = metaNumber(frame, "slope_current");
  const ratioMc = metaNumber(frame, "error_ratio_vs_mc");
  const sigma = metaNumber(frame, "sigma_measured");
  const bandAvail = frame.meta["band_available"] === true;

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = columnsLayout(scene, width, height, { topShare: 0.58, auxH: 150, gap: 48 });
  const main = makeAxis("main", [L.left, L.topY, L.leftW, L.topH], {
    xlim: [0, 1],
    ylim: [0, 1],
    xlabel: "$x_1$",
    ylabel: axisYlabelOf(SPEC?.axes, "main", "$x_2$"),
    title: axisTitleOf(SPEC?.axes, "main", "点集（单位正方形；$x_1$–$x_2$ 平面；蓝=MC / 橙=Sobol' / 绿=格点）"),
    autoscaled: false,
  });
  const aux = makeAxis("aux", [L.left, L.botY, L.leftW, L.botH], {
    xlim: [nAxis[0] as number, nAxis[nAxis.length - 1] as number],
    ylim: [-10, 0],
    xscale: "log",
    xlabel: "$N$（对数）",
    ylabel: axisYlabelOf(SPEC?.axes, "aux", "$\\log_{10}$ 误差"),
    title: axisTitleOf(SPEC?.axes, "aux", "$\\log_{10}$ 误差 vs $N$（参考 $\\sigma/\\sqrt{N}$）"),
    autoscaled: true,
  });
  scene.axes.push(main, aux);

  scene.layers.push(
    scatterLayer("main", mcX, mcY, { r: 2.2, color: pal.curves[0] as string, label: "MC" }),
    scatterLayer("main", sbX, sbY, { r: 2.6, color: pal.exact, label: "Sobol'" }),
    scatterLayer("main", laX, laY, { r: 2.2, color: pal.curves[1] as string, label: "秩一格点" }),
  );

  // 误差带先画（背景），有效点不足 2 个时整块不画——占位值不是误差。
  if (bandLo !== undefined && bandHi !== undefined && bandValid !== undefined) {
    const bx: number[] = [];
    const blo: number[] = [];
    const bhi: number[] = [];
    for (let i = 0; i < nAxis.length; i += 1) {
      if ((bandValid[i] as number) > 0.5 && Number.isFinite(bandLo[i] as number) && Number.isFinite(bandHi[i] as number)) {
        bx.push(nAxis[i] as number);
        blo.push(bandLo[i] as number);
        bhi.push(bandHi[i] as number);
      }
    }
    if (bx.length >= 2) {
      scene.layers.push(bandLayer("aux", { x: bx, ylo: blo, yhi: bhi, color: pal.exact, alpha: 0.22, label: "$R$ 个随机化复制的均值标准误带" }));
    }
  }
  const curveDefs = [
    { name: "mc", arr: errMc, color: pal.curves[0] as string },
    { name: "sobol", arr: errSb, color: pal.exact },
    { name: "lattice", arr: errLa, color: pal.curves[1] as string },
    { name: "grid", arr: errGr, color: pal.residual },
  ];
  for (const d of curveDefs) {
    scene.layers.push(
      lineLayer("aux", {
        x: nAxis,
        y: d.arr,
        color: d.color,
        width: d.name === method ? 2.4 : 1.2,
        dash: d.name === method ? undefined : [2, 3],
      }),
    );
  }
  scene.layers.push(lineLayer("aux", { x: nAxis, y: errRef, color: pal.reference, width: 1.3, dash: [5, 3] }));

  const budget = unitBudget(L.rightW);
  if (L.rightW >= 200) {
    let y = L.topY;
    const push = (lines: readonly string[], color: string): void => {
      const block = textBlock(L.rightX, y, L.floor, lines, color, budget, "top");
      scene.texts.push(...block);
      y += block.length * TEXT_ROW_H;
    };
    push(
      [
        `$k=${formatNumber(nShown, 0)}/${formatNumber(nMax, 0)}$（当前 ${methodName}）`,
        `被积函数：${integrandName}`,
        ...(integrandExpr !== "" ? [`f = ${integrandExpr}`] : []),
        sigma !== undefined
          ? `$\\sigma=${formatNumber(sigma, 3)}$；参考斜率 $s=${formatNumber(slopeRef ?? Number.NaN, 3)}$。`
          : `参考斜率 $s=${formatNumber(slopeRef ?? Number.NaN, 3)}$。`,
      ],
      pal.ink,
    );
    push(
      [
        "蓝=MC，橙=Sobol'，绿=格点，紫=张量格点（$x_1,x_2\\in[0,1)$）。",
        "灰虚=σ/√N；只对 MC 是理论（$\\sigma/\\sqrt{N}$ 的斜率是 $-1/2$）。",
      ],
      pal.muted,
    );
    push(
      [
        `斜率（$\\log$–$\\log$）：MC ${formatNumber(slopes.mc ?? Number.NaN, 3)}、Sobol ${formatNumber(slopes.sobol ?? Number.NaN, 3)}`,
        `格点 ${formatNumber(slopes.lattice ?? Number.NaN, 3)}、张量格点 ${formatNumber(slopes.grid ?? Number.NaN, 3)}。`,
        `当前方法 ${methodName} 的斜率 ${formatNumber(slopeSel ?? Number.NaN, 3)}；`,
        `$N$ 最大时误差是 MC 的 ${formatNumber(ratioMc ?? Number.NaN, 2)}$\\times$。`,
      ],
      pal.muted,
    );
    push(
      bandAvail && bandValid !== undefined
        ? [
            "误差带：$R$ 个随机化复制均值的标准误",
            "（对数轴上下沿），是可用的误差估计。",
            "参考线 $\\sigma/\\sqrt{N}$ 只对 MC 是理论；",
            "QMC 更陡是事实，但别外推到无穷 $N$。",
            "左图点集的“均匀”与右图的收敛率是两件事；",
            "均匀性要用 $D_N^{*}$ 量。",
          ]
        : [
            "误差带不存在（band_valid=0）：确定性复制",
            "的标准差恒为 $0$，$0$ 不等于误差小。",
            "参考线 $\\sigma/\\sqrt{N}$ 只对 MC 是理论；",
            "QMC 更陡是事实，但别外推到无穷 $N$。",
            "左图点集的“均匀”与右图的收敛率是两件事；",
            "均匀性要用 $D_N^{*}$ 量。",
          ],
      pal.muted,
    );
  } else {
    degraded.push("右侧说明列过窄：斜率读数并入角标");
  }

  scene.badges.push(placeBadge(sourceBadge(dataSourceOf(frame)), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge(
        "status",
        `k=${formatNumber(nShown, 0)}｜${methodName}｜slope=${formatNumber(slopeSel ?? Number.NaN, 2)}｜带${bandAvail ? "有" : "无"}`,
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
