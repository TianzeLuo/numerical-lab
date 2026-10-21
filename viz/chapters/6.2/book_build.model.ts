/**
 * `book_build`（6.2 章 viz 3）：讲义构建的逐章审计矩阵。
 *
 * 六列检查（行=章）：
 *   笔记（存在且含 theorem） / 附表（行数=断言报告条数） /
 *   引用（章内闭合、无重复 label） / 输入（\input 目标存在） /
 *   字形（notes+附表无 xelatex 无字形风险字符） / 陷阱（pitfalls ≥800 字）。
 *
 * 热图 1=通过、0=缺；副图是每章的问题数（0..6）。纵向留 0.5 的空白只为让 0 值点可见，
 * 不隐藏任何负值（问题数没有负值）。
 */

import { dataToPixel } from "../../shared/src/geometry.js";
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
  sceneShell,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  heatLayer,
  makeAxis,
  scatterLayer,
  sourceBadge,
  text,
} from "../../shared/src/scene.js";
import { axisTitleOf, axisYlabelOf, columnsLayout } from "../_common/panel.js";
import {
  CHAPTER,
  HEAT_RIGHT_GUTTER,
  SLOT,
  VIZ_TITLES,
  compactText,
  fmt,
  jsonList,
  narrowLayout,
  pushGroups,
  statusLine,
} from "./common.js";
import { manifest } from "./manifest.gen.js";

const VIZ = "book_build";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? VIZ_TITLES.book_build;

export const bookBuildModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);
  const values = requireArray(frame, "check_values");
  const issues = requireArray(frame, "error_y");
  const tableRows = requireArray(frame, "table_rows");
  const declared = requireArray(frame, "declared");
  const labels = jsonList(frame.strings["chapter_ids"]);
  const checkLabels = jsonList(frame.strings["check_labels"]);
  const n = issues.length;
  if (labels.length !== n) throw new Error(`book_build：${n} 行但 chapter_ids=${labels.length}`);
  if (checkLabels.length !== 6) throw new Error(`book_build：check_labels 应有 6 项，收到 ${checkLabels.length}`);
  if (values.length !== n * 6) throw new Error(`book_build：check_values 长度 ${values.length} ≠ ${n}×6`);
  if (tableRows.length !== n || declared.length !== n) throw new Error("book_build：附表/声明数组长度不一致");
  const pitfalls = requirePitfalls(frame);

  const nIssues = metaNumber(frame, "n_issues") ?? Array.from(issues).reduce((a, b) => a + b, 0);
  const sort = frame.strings["sort"] ?? "topo";
  const part = frame.strings["part"] ?? "all";
  const summary = frame.strings["summary"] ?? "";
  const tableNote = frame.strings["table_note"] ?? "";
  const refNote = frame.strings["ref_note"] ?? "";
  const glyphNote = frame.strings["glyph_note"] ?? "";
  const pitfallNote = frame.strings["pitfall_note"] ?? "";
  const issueNote = frame.strings["issue_note"] ?? "";
  const checkNote = frame.strings["check_note"] ?? "";

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: 0,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });
  const degraded: string[] = [];
  const wide = width >= 820;

  let heatRect: [number, number, number, number];
  let auxRect: [number, number, number, number] | null = null;
  let sidebar: { x: number; w: number } | null = null;
  if (wide) {
    const L = columnsLayout(scene, width, height, { topShare: 0.7, auxH: 104, gap: 56 });
    heatRect = [L.left, L.topY, Math.max(150, L.leftW - HEAT_RIGHT_GUTTER), L.topH];
    auxRect = [L.left, L.botY, Math.max(150, L.leftW - HEAT_RIGHT_GUTTER), Math.min(104, L.botH)];
    sidebar = { x: L.rightX, w: L.rightW };
  } else {
    const N = narrowLayout(scene, width, height, { textRows: 3, mainShare: 1, aux: false });
    heatRect = [N.left, N.main[1], Math.max(150, N.contentW - HEAT_RIGHT_GUTTER), N.main[3]];
    compactText(
      scene,
      N.left,
      N.textTop,
      N.contentW,
      [
        `${fmt(n, 0)} 章 × 6 项；问题格 ${fmt(nIssues, 0)}；排序 ${sort}。`,
        tableNote,
        "列：笔记/附表/引用/输入/字形/陷阱；$1$=通过。",
      ],
      nIssues > 0 ? pal.warn : pal.muted,
      N.main[1] - 2,
    );
    degraded.push("窄画布：省略问题散点与右栏说明；热图与行列标签保留");
  }

  const main = makeAxis("main", heatRect, {
    xlim: [0, 5],
    ylim: [0, Math.max(1, n - 1)],
    xlabel: "",
    ylabel: axisYlabelOf(SPEC?.axes, "main", "章（行 $0$ 在底部）"),
    title: axisTitleOf(SPEC?.axes, "main", "逐章构建检查（$1$=通过，$0$=缺）"),
    xticks: [],
    yticks: [],
    grid: false,
    autoscaled: false,
  });
  scene.axes.push(main);

  // 热图约定：row 0 在 y-min；显示行 0（目录第一）放到最后一行。
  const heat: number[] = new Array(n * 6).fill(0);
  for (let i = 0; i < n; i += 1) {
    for (let c = 0; c < 6; c += 1) {
      heat[(n - 1 - i) * 6 + c] = values[i * 6 + c] as number;
    }
  }
  scene.layers.push(
    heatLayer("main", heat, 6, n, [0, 5, 0, Math.max(1, n - 1)], {
      cmap: "viridis",
      vmin: 0,
      vmax: 1,
      logScale: false,
      label: "构建检查（$1$=通过）",
    }),
  );

  // 行标签：行距只有约 8–10px，按 1/3 抽稀；失败行必须出现，
  // 出现时把相邻的普通标签让开（否则两个 14px 高的文本框会重叠）。
  const stride = n <= 12 ? 1 : 3;
  const chosen = new Set<number>();
  for (let i = 0; i < n; i += stride) chosen.add(i);
  chosen.add(n - 1);
  for (let i = 0; i < n; i += 1) {
    if ((issues[i] as number) <= 0) continue;
    if (!chosen.has(i)) {
      chosen.delete(i - 1);
      chosen.delete(i + 1);
      chosen.add(i);
    }
  }
  for (const i of Array.from(chosen).sort((a, b) => a - b)) {
    const failing = (issues[i] as number) > 0;
    const [, py] = dataToPixel(main, 0, Math.max(0, n - 1 - i));
    scene.texts.push(
      text(heatRect[0] - 4, py - 5, labels[i] as string, {
        role: "label",
        size: 11,
        color: failing ? pal.warn : pal.muted,
        align: "right",
      }),
    );
  }
  for (let c = 0; c < 6; c += 1) {
    const [px] = dataToPixel(main, c, 0);
    scene.texts.push(
      text(px, heatRect[1] + heatRect[3] + 18, checkLabels[c] as string, {
        role: "axis",
        size: 11,
        color: pal.muted,
        align: "center",
      }),
    );
  }

  if (auxRect) {
    const maxIssues = Math.max(1, ...Array.from(issues));
    const error = makeAxis("error", auxRect, {
      xlim: [0, Math.max(1, n - 1)],
      ylim: [-0.5, maxIssues * 1.12],
      xlabel: "$i$（当前排序）",
      ylabel: axisYlabelOf(SPEC?.axes, "error", "问题数（$0$ 到 $6$）"),
      title: axisTitleOf(SPEC?.axes, "error", "每章问题数（$0$=通过；下留白 $0.5$ 仅为显示 $0$ 点）"),
      xticks: [],
      yticks: Array.from({ length: Math.floor(maxIssues) + 1 }, (_, i) => i),
      grid: false,
      autoscaled: false,
    });
    scene.axes.push(error);
    const okX: number[] = [];
    const okY: number[] = [];
    const badX: number[] = [];
    const badY: number[] = [];
    for (let i = 0; i < n; i += 1) {
      if ((issues[i] as number) > 0) {
        badX.push(i);
        badY.push(issues[i] as number);
      } else {
        okX.push(i);
        okY.push(0);
      }
    }
    if (okX.length > 0) scene.layers.push(scatterLayer("error", okX, okY, { r: 3.2, color: pal.curves[1] as string, alpha: 1 }));
    if (badX.length > 0) scene.layers.push(scatterLayer("error", badX, badY, { r: 4.2, color: pal.warn, alpha: 1 }));
  }

  if (sidebar) {
    const bottom = auxRect ? auxRect[1] + auxRect[3] : heatRect[1] + heatRect[3];
    pushGroups(scene, sidebar.x, heatRect[1], bottom, sidebar.w, [
      { lines: [summary, checkNote], color: pal.ink },
      { lines: [tableNote, refNote, glyphNote, pitfallNote], color: pal.muted },
      { lines: [issueNote], color: nIssues > 0 ? pal.warn : pal.curves[1] as string },
      { lines: [`排序 ${sort}；分部 ${part}。`, "行=章；列=检查项；$1$=通过。", "问题数 $=6-\\sum_j v_{ij}$。"], color: pal.muted },
    ]);
  }

  let sumTable = 0;
  let sumDeclared = 0;
  for (let i = 0; i < n; i += 1) {
    sumTable += tableRows[i] as number;
    sumDeclared += declared[i] as number;
  }
  scene.badges.push(placeBadge(sourceBadge(dataSourceOf(frame)), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge(
        "status",
        statusLine([
          `${fmt(n, 0)} 章 × 6`,
          `问题 ${fmt(nIssues, 0)}`,
          `附表 ${fmt(sumTable, 0)}/${fmt(sumDeclared, 0)}`,
          sort,
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
