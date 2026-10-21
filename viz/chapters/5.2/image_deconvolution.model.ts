/**
 * `image_deconvolution`（5.2 章 viz 3）：24×24 幻影在模糊 + 噪声下的四联重建。
 *
 * 四块热图是**四个不同的量**，不是“同一张图的四种滤镜”：真解 x*、带噪数据 b、
 * 当前方法的重建 x_k、以及 |x_k − x*|（后者只有仿真才知道）。每块热图有各自的
 * 色标：光栅器（`tools/raster.py`）在每个 heat 轴的右侧自动画色标，所以版式必须
 * 为它留出 120px 的过道——这条纪律在 1.4/3.2/4.4 已经踩过坑，这里不再复述。
 *
 * 图像的行 0 在数组里是**顶行**（`numcore.inverse.regularize.phantom_image`
 * 用 mgrid 生成），而 heat 图层把行 0 画在轴底。所有热图统一翻转一次，
 * 屏幕上的上下才与 `np.mgrid` 一致；不翻不是“风格不同”，是**上下颠倒**。
 *
 * 历史轴上的误差与残差是两条不同的曲线：残差小 ≠ 解准（半收敛时尤其如此）。
 * 谱方法（tikhonov/tsvd）的残差历史写地板而不是 0；日志游标 k 只是截断阶族的
 * 位置，**不进入重建**——这两件事都必须明说。
 *
 * 帧 `time.mode="scrub"`：`scene.t` 取帧 `meta.t`，`k_current` 是迭代/截断阶。
 */

import { formatNumber } from "../../shared/src/axes.js";
import {
  applyOverflow,
  canvasSize,
  dataSourceOf,
  degradedBadge,
  finiteMinMax,
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
  heatLayer,
  lineLayer,
  makeAxis,
  scatterLayer,
  sourceBadge,
} from "../../shared/src/scene.js";
import {
  TEXT_ROW_H,
  columnsLayout,
  positiveMinMax,
  textBlock,
  unitBudget,
} from "../_common/panel.js";
import { CHAPTER, SLOT, nearestIndex } from "./common.js";
import { manifest } from "./manifest.gen.js";

const VIZ = "image_deconvolution";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? "图像反卷积：噪声什么时候把解吞掉";

type ImageKey = "image_true" | "image_recon" | "image_abs_error";

/** heat 轴右侧的色标过道：轴右 46px 起画宽 14px 的条，再留数值标签的宽度。 */
const COLORBAR_GUTTER = 120;

/**
 * 行主序 `rows × cols` 的数组 → 行 0 在顶的显示缓冲。
 *
 * heat 渲染把行 0 放在 extent 底部（`render2d.ts` 的快慢两条路径都如此），
 * 所以图像类数据必须在这里翻一次；`rowOfMatrix` 之类的“取某一行”不受影响。
 */
function flipRows(values: ArrayLike<number>, rows: number, cols: number): Float64Array {
  const out = new Float64Array(values.length);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      out[(rows - 1 - r) * cols + c] = values[r * cols + c] as number;
    }
  }
  return out;
}

export const imageDeconvolutionModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);

  const n = metaNumber(frame, "n_image");
  if (n === undefined || !Number.isInteger(n) || n <= 0) {
    throw new Error("帧 meta 缺 n_image（正整数）：热图的 nx/ny 不能猜");
  }
  const operator = stringOf(frame, "operator") ?? "?";
  const method = stringOf(frame, "method") ?? "?";
  const histKind = stringOf(frame, "history_kind") ?? "?";
  const nAngles = metaNumber(frame, "n_angles");

  const imageTrue = requireArray(frame, "image_true");
  const imageData = requireArray(frame, "image_data");
  const imageRecon = requireArray(frame, "image_recon");
  const imageAbs = requireArray(frame, "image_abs_error");
  for (const [key, arr] of [
    ["image_true", imageTrue],
    ["image_recon", imageRecon],
    ["image_abs_error", imageAbs],
  ] as const) {
    if (arr.length !== n * n) throw new Error(`${key} 长度 ${arr.length} ≠ n_image²=${n * n}`);
  }

  // CT 的数据是正弦图（na × (na+2)），不是 n×n；按长度与 meta.n_angles 显式判定。
  let dataRows = n;
  let dataCols = n;
  if (imageData.length !== n * n) {
    const na = nAngles !== undefined ? Math.round(nAngles) : -1;
    if (operator === "ct" && na > 0 && imageData.length === na * (na + 2)) {
      dataRows = na;
      dataCols = na + 2;
    } else {
      throw new Error(
        `image_data 长度 ${imageData.length} 既不是 n²=${n * n}，也不是 CT 正弦图的 na(na+2)` +
          `（operator=${operator}, n_angles=${nAngles ?? "缺"}）：不猜数据形状`,
      );
    }
  }

  const histK = requireArray(frame, "history_k");
  const errHist = requireArray(frame, "error_history");
  const resHist = requireArray(frame, "residual_history");
  requireSameLength(histK, errHist, "history_k 与 error_history");
  requireSameLength(histK, resHist, "history_k 与 residual_history");
  const pitfalls = requirePitfalls(frame);

  const hasSweep = frame.has("log10_lam_sweep") && frame.has("error_lam_sweep");
  const lgSweep = hasSweep ? requireArray(frame, "log10_lam_sweep") : undefined;
  const errSweep = hasSweep ? requireArray(frame, "error_lam_sweep") : undefined;
  if (lgSweep !== undefined && errSweep !== undefined) {
    requireSameLength(lgSweep, errSweep, "log10_lam_sweep 与 error_lam_sweep");
  }
  const degraded: string[] = [];
  if (!hasSweep) degraded.push("帧缺 λ 扫描数组（迭代法没有 λ 扫描）：不画误差-λ 曲线");

  const t = metaNumber(frame, "t") ?? opts.t;
  const kCurrentMeta = metaNumber(frame, "k_current");
  const kMax = Math.max(0, histK.length - 1);
  const kCurrent = Math.max(0, Math.min(kMax, kCurrentMeta ?? Math.round(t)));
  const floorMeta = metaNumber(frame, "floor");
  const floor = floorMeta !== undefined && floorMeta > 0 ? floorMeta : 1e-17;
  if (floorMeta === undefined) degraded.push("帧 meta 缺 floor：地板用 1e-17");

  const directMethod = method === "tikhonov" || method === "tsvd";
  const kappa = metaNumber(frame, "kappa");
  const kappa1d = metaNumber(frame, "kappa_1d");
  const noiseRel = metaNumber(frame, "noise_rel");
  const noiseModes = metaNumber(frame, "noise_dominated_modes");
  const nMod = metaNumber(frame, "n_modes");
  const rank = metaNumber(frame, "rank_measured");
  const nullDim = metaNumber(frame, "null_space_dim");
  const errNow = metaNumber(frame, "error_rel_current");
  const resNow = metaNumber(frame, "residual_rel_current");
  const errBest = metaNumber(frame, "error_rel_best_lambda");
  const lamBest = metaNumber(frame, "lam_best");
  const kBest = metaNumber(frame, "k_best");
  const errAtKBest = metaNumber(frame, "error_rel_at_k_best");
  const semi = frame.meta["semi_convergence"] === true;
  const lgLam = metaNumber(frame, "log10_lam");

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });

  const L = columnsLayout(scene, width, height, { topShare: 0.5, auxH: 140, gap: 46 });
  const roomy = width >= 900;

  // 各热图的独立色标范围。真解固定 [0,1]（幻影只有 0/0.5/0.75/1 四档）；
  // 数据与重建按自己的有限值取窗，绝不把 CT 正弦图硬塞进 [0,1]。
  const trueMM = finiteMinMax(imageTrue) ?? [0, 1];
  const dataMM = finiteMinMax(imageData) ?? [0, 1];
  const reconMM = finiteMinMax(imageRecon) ?? [0, 1];
  const absMax = Math.max(1e-12, finiteMinMax(imageAbs)?.[1] ?? 1);
  const pad = (lo: number, hi: number): [number, number] => {
    if (!(hi > lo)) return [lo, lo + 1];
    const d = 0.04 * (hi - lo);
    return [lo - d, hi + d];
  };
  const [dataLo, dataHi] = pad(Math.min(0, dataMM[0]), Math.max(1e-12, dataMM[1]));
  const [reconLo, reconHi] = pad(reconMM[0], reconMM[1]);
  // 幻影只有 0/0.5/0.75/1 四档：真解图固定 [0,1]，不让“留白”把 0 画成非零色。
  const trueLo = Math.min(0, trueMM[0]);
  const trueHi = Math.max(1, trueMM[1]);

  interface ThumbDef {
    key: ImageKey | "image_data";
    title: string;
    cmap: "viridis" | "magma";
    vmin: number;
    vmax: number;
    rows: number;
    cols: number;
  }
  const dataTitle = dataRows === dataCols ? "数据 b" : "正弦图 b";
  const thumbs: ThumbDef[] = [
    { key: "image_true", title: "真解 x*", cmap: "viridis", vmin: trueLo, vmax: trueHi, rows: n, cols: n },
    { key: "image_data", title: dataTitle, cmap: "viridis", vmin: dataLo, vmax: dataHi, rows: dataRows, cols: dataCols },
    { key: "image_recon", title: "重建 x_k", cmap: "viridis", vmin: reconLo, vmax: reconHi, rows: n, cols: n },
    { key: "image_abs_error", title: "|重建−x*|", cmap: "magma", vmin: 0, vmax: absMax, rows: n, cols: n },
  ];
  const drawThumbs = roomy ? thumbs : thumbs.slice(2); // 窄画布只画重建与误差

  let gridBottom = L.topY;
  let narrowHistTop = L.topY;
  let narrowHistH = 0;
  const narrowInnerW = Math.max(240, width - L.left - 24);
  const narrowHeatW = Math.max(90, Math.min(160, Math.floor(narrowInnerW * 0.28)));
  if (roomy) {
    const avail = Math.max(240, L.floor - L.topY);
    const heatH = Math.max(92, Math.min(122, Math.floor((avail - 44 - 3 * TEXT_ROW_H - 6) / 2)));
    const thumbW = Math.max(96, Math.floor((L.leftW - 2 * COLORBAR_GUTTER - 14) / 2));
    const colX = [L.left, L.left + thumbW + COLORBAR_GUTTER + 14];
    const rowY = [L.topY, L.topY + heatH + 44];
    drawThumbs.forEach((d, i) => {
      const x = colX[i % 2] as number;
      const y = rowY[Math.floor(i / 2)] as number;
      const ax = makeAxis(`img_${i}`, [x, y, thumbW, heatH], {
        xlim: [0, d.cols],
        ylim: [0, d.rows],
        xlabel: "",
        ylabel: "",
        title: d.title,
        autoscaled: false,
        xticks: [],
        yticks: [],
      });
      scene.axes.push(ax);
      const raw = d.key === "image_data" ? imageData : (d.key === "image_true" ? imageTrue : d.key === "image_recon" ? imageRecon : imageAbs);
      // label 留空：标题已经说了这块是什么，再让光栅器在色标下写一遍只会挤占过道。
      scene.layers.push(
        heatLayer(ax.id, flipRows(raw, d.rows, d.cols), d.cols, d.rows, [0, d.cols, 0, d.rows], {
          cmap: d.cmap,
          vmin: d.vmin,
          vmax: d.vmax,
        }),
      );
    });
    gridBottom = (rowY[1] as number) + heatH;
  } else {
    // 窄画布（640 级）：徽标区已占去 8 行，只放得下一块重建热图 + 历史轴。
    // 轴标题画在 rect 上方 20px，因此把面板整体下移 18px，避开最后一排徽标。
    const top = L.topY + 18;
    const avail = Math.max(100, L.floor - top);
    const heatH = Math.max(72, Math.min(100, avail - 38));
    const recon = thumbs[2] as ThumbDef;
    const ax = makeAxis("img_0", [L.left, top, narrowHeatW, heatH], {
      xlim: [0, recon.cols],
      ylim: [0, recon.rows],
      xlabel: "",
      ylabel: "",
      title: recon.title,
      autoscaled: false,
      xticks: [],
      yticks: [],
    });
    scene.axes.push(ax);
    scene.layers.push(
      heatLayer(ax.id, flipRows(imageRecon, recon.rows, recon.cols), recon.cols, recon.rows, [0, recon.cols, 0, recon.rows], {
        cmap: recon.cmap,
        vmin: recon.vmin,
        vmax: recon.vmax,
      }),
    );
    gridBottom = top + heatH;
    narrowHistTop = top;
    narrowHistH = heatH;
    degraded.push("窄画布：只画重建热图与历史（真解/数据/误差图与 λ 扫描省略）");
  }

  // —— 历史轴：误差与残差分开；不画内联标签，避免在曲线末端互相压住 ----------
  const rightAvail = Math.max(120, L.floor - L.topY);
  const rightHistH = roomy ? Math.max(100, Math.min(132, Math.round(rightAvail * 0.36))) : narrowHistH;
  const histTop = roomy ? L.topY : narrowHistTop;
  const histH = roomy ? rightHistH : narrowHistH;
  const histX = roomy ? L.rightX : L.left + narrowHeatW + COLORBAR_GUTTER + 14;
  const histW = roomy ? L.rightW : Math.max(120, width - histX - 24);
  // 谱方法的 residual_history 是“没有定义 → 后端写 0 → 地板值”，把它画进对数轴会把
  // 误差曲线压成顶部一条直线。这类方法不画残差与地板线，并在正文里点名，
  // 轴窗只包住误差曲线本身（下限留半个数量级的地板）。
  const residualMeaningful = histKind === "iteration";
  const errPos = positiveMinMax(errHist);
  const resPos = positiveMinMax(resHist);
  const histMin = residualMeaningful ? Math.min(errPos?.[0] ?? 1, resPos?.[0] ?? 1) : (errPos?.[0] ?? 1);
  const histLo = Math.max(floor, histMin * 0.5);
  let histAxis: ReturnType<typeof makeAxis> | undefined;
  if (histH >= 60) {
    const errHi = finiteMinMax(errHist)?.[1] ?? 1;
    histAxis = makeAxis("history", [histX, histTop, histW, histH], {
      xlim: [0, Math.max(1, kMax)],
      ylim: [histLo, Math.max(histLo * 10, errHi * 1.5)],
      yscale: "log",
      xlabel: histKind === "iteration" ? "迭代步 k" : "截断阶 k",
      ylabel: residualMeaningful ? "相对量（对数）" : "相对误差（对数）",
      title: residualMeaningful ? "误差（红）/ 残差（蓝）随 k" : "误差随截断阶 k（红）",
      autoscaled: true,
    });
    scene.axes.push(histAxis);
    scene.layers.push(lineLayer("history", { x: histK, y: errHist, color: pal.marker, width: 2, label: "重建误差" }));
    if (residualMeaningful) {
      scene.layers.push(lineLayer("history", { x: histK, y: resHist, color: pal.residual, width: 1.6, label: "残差" }));
      if (floor >= histLo && floor <= (histAxis.ylim[1] as number)) {
        scene.layers.push(
          lineLayer("history", { x: [histAxis.xlim[0], histAxis.xlim[1]], y: [floor, floor], color: pal.warn, width: 1, dash: [3, 3], label: "误差地板" }),
        );
      }
    }
    if (kBest !== undefined && kBest >= 0 && kBest <= kMax) {
      scene.layers.push(
        lineLayer("history", {
          x: [kBest, kBest],
          y: [histAxis.ylim[0], histAxis.ylim[1]],
          color: pal.exact,
          width: 1.2,
          dash: [4, 3],
        }),
      );
      if (errAtKBest !== undefined) {
        scene.layers.push(scatterLayer("history", [kBest], [errAtKBest], { r: 3.6, color: pal.exact, label: "最佳截断阶" }));
      }
    }
    scene.layers.push(
      lineLayer("history", {
        x: [kCurrent, kCurrent],
        y: [histAxis.ylim[0], histAxis.ylim[1]],
        color: pal.cursor,
        width: 1.2,
        dash: [2, 2],
      }),
    );
    if (kCurrent >= 0 && kCurrent <= kMax) {
      scene.layers.push(scatterLayer("history", [kCurrent], [errHist[kCurrent] as number], { r: 3.4, color: pal.marker, label: "当前误差" }));
    }
  } else {
    degraded.push("画布过矮：历史轴放不下（<60px），只保留热图");
  }

  // —— 右上：λ 扫描（只有直接法有）-------------------------------------------
  const sweepTop = L.topY + rightHistH + 60;
  const sweepH = Math.max(64, Math.min(96, rightAvail - rightHistH - 29 - 60 - 30));
  let sweepAxis: ReturnType<typeof makeAxis> | undefined;
  if (roomy && lgSweep !== undefined && errSweep !== undefined && lgSweep.length >= 2) {
    const sweepPos = positiveMinMax(errSweep);
    const sweepLo = Math.max(floor, (sweepPos?.[0] ?? 1) * 0.5);
    const sweepHi = Math.max(sweepLo * 10, (finiteMinMax(errSweep)?.[1] ?? 1) * 1.5);
    sweepAxis = makeAxis("lambda_sweep", [L.rightX, sweepTop, L.rightW, sweepH], {
      xlim: [lgSweep[0] as number, lgSweep[lgSweep.length - 1] as number],
      ylim: [sweepLo, sweepHi],
      yscale: "log",
      xlabel: "$\\log_{10}\\lambda$",
      ylabel: "相对误差",
      title: "误差随 $\\lambda$（两端失败）",
      autoscaled: true,
    });
    scene.axes.push(sweepAxis);
    scene.layers.push(lineLayer("lambda_sweep", { x: lgSweep, y: errSweep, color: pal.curves[0] as string, width: 1.8, label: "误差随 $λ$" }));
    if (lamBest !== undefined && lamBest > 0) {
      const lgBest = Math.log10(lamBest);
      if (lgBest >= sweepAxis.xlim[0] && lgBest <= sweepAxis.xlim[1]) {
        scene.layers.push(
          lineLayer("lambda_sweep", {
            x: [lgBest, lgBest],
            y: [sweepAxis.ylim[0], sweepAxis.ylim[1]],
            color: pal.exact,
            width: 1.1,
            dash: [4, 3],
          }),
        );
      }
    }
    if (lgLam !== undefined) {
      const j = nearestIndex(lgSweep, lgLam);
      scene.layers.push(
        lineLayer("lambda_sweep", {
          x: [lgLam, lgLam],
          y: [sweepAxis.ylim[0], sweepAxis.ylim[1]],
          color: pal.marker,
          width: 1.2,
          dash: [2, 2],
        }),
      );
      if (j >= 0) scene.layers.push(scatterLayer("lambda_sweep", [lgLam], [errSweep[j] as number], { r: 3.2, color: pal.marker, label: "当前 $λ$ 的误差" }));
    }
  } else if (roomy && histAxis === undefined) {
    degraded.push("右列没有可画的历史轴：λ 扫描也不画");
  }

  // —— 文字：宽画布时宽行说明在热图下方，右列只放读数 -------------------------
  const lamText = directMethod
    ? `λ=10^${formatNumber(lgLam ?? Number.NaN, 2)}`
    : method === "tv"
      ? `λ=10^${formatNumber(lgLam ?? Number.NaN, 2)}（TV 权重）`
      : "λ 不参与（迭代早停）";
  const methodLine = directMethod
    ? "直接法：k 不进入重建（图像由 λ 定）；残差历史写地板，不画在轴上。"
    : "迭代法：k 就是正则化强度；残差小 ≠ 解准。";
  const semiLine = semi
    ? `半收敛：误差先降后升，k*=${formatNumber(kBest ?? Number.NaN, 0)}。`
    : "没有检测到半收敛（误差单调或已到地板）。";
  const sweepSemiLine =
    hasSweep && lamBest !== undefined && errBest !== undefined
      ? `λ 最优 ${formatNumber(lamBest, 3)}；${semiLine}`
      : semiLine;

  if (roomy) {
    const noteTop = gridBottom + 4;
    const budget = unitBudget(L.leftW);
    // 逐个 textBlock 共享同一条起始 y：用已画行数下推，而不是让每个块各自从 noteTop 开始。
    let y = noteTop;
    const line = (body: string, color: string): void => {
      const block = textBlock(L.left, y, L.floor, [body], color, budget, "top");
      scene.texts.push(...block);
      y += block.length * TEXT_ROW_H;
    };
    line(
      `${method} @ ${operator}｜k=${kCurrent}/${kMax}｜${lamText}｜误差 ${formatNumber(errNow ?? Number.NaN, 3)}，残差 ${formatNumber(resNow ?? Number.NaN, 3)}。`,
      pal.ink,
    );
    line(methodLine, pal.muted);
    line(sweepSemiLine, semi ? pal.warn : pal.muted);
    if (Math.floor((L.floor - y) / TEXT_ROW_H) >= 1) {
      line("竖线：k 图橙虚线=k*、灰点线=当前 k；λ 图橙虚线=扫描最优、红点线=当前 λ。", pal.muted);
    }

    // 右列读数（预算 41 单位，短行不折；只画放得下的整行，宁缺勿截）。
    const readTop = sweepAxis !== undefined ? sweepAxis.rect[1] + sweepAxis.rect[3] + 34 : histTop + histH + 34;
    const rows = Math.max(0, Math.floor((L.floor - readTop) / TEXT_ROW_H));
    if (rows >= 1) {
      const rbudget = unitBudget(L.rightW);
      const readings = [
        {
          body:
            kappa1d !== undefined && Number.isFinite(kappa1d)
              ? `κ₂D=${formatNumber(kappa ?? Number.NaN, 2)}，κ₁D=${formatNumber(kappa1d, 2)}`
              : `κ₂D=${formatNumber(kappa ?? Number.NaN, 2)}（CT 无 κ₁D）`,
          color: pal.ink,
        },
        {
          body: `秩 ${formatNumber(rank ?? Number.NaN, 0)}/${formatNumber(nMod ?? Number.NaN, 0)}，零空间 ${formatNumber(nullDim ?? Number.NaN, 0)}`,
          color: pal.muted,
        },
        {
          body: `噪声接管 ${formatNumber(noiseModes ?? Number.NaN, 0)} 个模态；噪声相对 ${formatNumber(noiseRel ?? Number.NaN, 2)}`,
          color: pal.muted,
        },
      ].slice(0, rows);
      readings.forEach((r, i) => {
        scene.texts.push(...textBlock(L.rightX, readTop + i * TEXT_ROW_H, L.floor, [r.body], r.color, rbudget, "top"));
      });
    }
  } else {
    // 窄画布：没有右列与文字区，把最要紧的两句塞到热图与历史之间。
    const budget = unitBudget(width - L.left - 24);
    const noteTop = histAxis !== undefined ? histAxis.rect[1] + histAxis.rect[3] + 34 : gridBottom + 24;
    if (L.floor - noteTop >= TEXT_ROW_H) {
      const block = textBlock(
        L.left,
        noteTop,
        L.floor,
        [`${method} @ ${operator}：${semi ? "半收敛" : "无半收敛"}${directMethod ? "；k 不进入重建" : "；k 即正则化强度"}。`],
        semi ? pal.warn : pal.muted,
        budget,
        "top",
      );
      scene.texts.push(...block);
    }
  }

  scene.badges.push(placeBadge(sourceBadge(dataSourceOf(frame)), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge("status", `${method}｜k=${kCurrent}/${kMax}｜err=${formatNumber(errNow ?? Number.NaN, 3)}｜res=${formatNumber(resNow ?? Number.NaN, 3)}`, 8, 8),
      SLOT.status,
      width,
    ),
  );
  if (noiseModes !== undefined && noiseModes > 0) {
    scene.badges.push(
      placeBadge(badge("critical", `噪声接管 ${formatNumber(noiseModes, 0)}/${formatNumber(nMod ?? Number.NaN, 0)} 个模态（救不回）`, 8, 8), SLOT.critical, width),
    );
  }
  const dbg = degradedBadge(degraded, SLOT.degraded, width);
  if (dbg !== undefined) scene.badges.push(dbg);

  applyOverflow(scene, 0, { slot: SLOT.overflow, width });
  requireDrawableLayers(scene);
  return scene;
};
