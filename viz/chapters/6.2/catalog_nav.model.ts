/**
 * `catalog_nav`（6.2 章 viz 1）：28 章的依赖图与构建顺序。
 *
 * 三条纪律
 * -------
 * 1. **目录不是章节号列表**：节点按最长路径分层；一条边 u→v 表示 u 必须先构建。
 * 2. **高亮不是删除**：选分部/焦点时，未选中的节点仍然画出来（否则跨部依赖会被藏掉）。
 * 3. **错误计数是四格账**：环 / 未知依赖 / 自依赖 / 缺模块，全部应为 0；副图直接画数。
 *
 * 帧没有时间轴（`time.mode="none"`），`opts.t` 恒为 0。
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
  barsLayer,
  lineLayer,
  makeAxis,
  scatterLayer,
  series,
  sourceBadge,
  text,
  type LineSeries,
} from "../../shared/src/scene.js";
import { axisTitleOf, axisYlabelOf, columnsLayout } from "../_common/panel.js";
import {
  CHAPTER,
  SLOT,
  VIZ_TITLES,
  compactText,
  fmt,
  jsonList,
  narrowLayout,
  palette as sharedPalette,
  partColors,
  pushGroups,
  statusLine,
} from "./common.js";
import { manifest } from "./manifest.gen.js";

const VIZ = "catalog_nav";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? VIZ_TITLES.catalog_nav;

const ERROR_LABELS = ["环", "未知依赖", "自依赖", "缺模块"] as const;

export const catalogNavModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);
  const nodeX = requireArray(frame, "node_x");
  const nodeY = requireArray(frame, "node_y");
  const nodePart = requireArray(frame, "node_part");
  const highlight = requireArray(frame, "node_highlight");
  const focusMask = requireArray(frame, "node_focus");
  const edgeX0 = requireArray(frame, "edge_x0");
  const edgeY0 = requireArray(frame, "edge_y0");
  const edgeX1 = requireArray(frame, "edge_x1");
  const edgeY1 = requireArray(frame, "edge_y1");
  const edgeHighlight = requireArray(frame, "edge_highlight");
  const errorValues = requireArray(frame, "error_values");
  const ids = jsonList(frame.strings["chapter_ids"]);
  const parts = jsonList(frame.strings["parts"]);
  const n = nodeX.length;
  if (ids.length !== n || parts.length !== n) {
    throw new Error(`catalog_nav：节点 ${n} 个，但 chapter_ids=${ids.length} / parts=${parts.length}`);
  }
  if (edgeX1.length !== edgeX0.length || edgeY1.length !== edgeY0.length || edgeHighlight.length !== edgeX0.length) {
    throw new Error("catalog_nav：边的四个坐标数组长度不一致");
  }
  const pitfalls = requirePitfalls(frame);

  const focus = frame.strings["focus"] ?? "?";
  const part = frame.strings["part"] ?? "all";
  const chain = frame.strings["chain"] ?? "?";
  const closureNote = frame.strings["closure_note"] ?? "";
  const problemNote = frame.strings["problem_note"] ?? "";
  const nEdges = edgeX0.length;
  const nProblems = metaNumber(frame, "n_violations") ?? Math.max(0, ...Array.from(errorValues));
  const maxDepth = metaNumber(frame, "max_depth") ?? Math.max(0, ...Array.from(nodeX));

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

  let mainRect: [number, number, number, number];
  let auxRect: [number, number, number, number];
  let sidebar: { x: number; w: number } | null = null;
  if (wide) {
    const L = columnsLayout(scene, width, height, { topShare: 0.71, auxH: 86, gap: 50 });
    mainRect = [L.left, L.topY, L.leftW, L.topH];
    auxRect = [L.left, L.botY, L.leftW, Math.min(118, L.botH)];
    sidebar = { x: L.rightX, w: L.rightW };
  } else {
    const N = narrowLayout(scene, width, height, { textRows: 2, mainShare: 0.64, aux: true });
    mainRect = N.main;
    auxRect = N.aux as [number, number, number, number];
    const lines = [
      `${fmt(n, 0)} 章 · ${fmt(nEdges, 0)} 条边 · ${fmt(metaNumber(frame, "n_layers"), 0)} 层；焦点 ${focus}。`,
      "颜色=分部；红圈=焦点章；边=前置依赖；未选中节点仍保留。",
    ];
    compactText(scene, N.left, N.textTop, N.contentW, lines, pal.muted, mainRect[1] - 2);
    degraded.push("窄画布：省略右栏说明；节点标签与副图保留");
  }

  const yPad = 0.75;
  const yLo = Math.min(...Array.from(nodeY)) - yPad;
  const yHi = Math.max(...Array.from(nodeY)) + yPad;
  const main = makeAxis("main", mainRect, {
    xlim: [-0.55, Math.max(1, maxDepth) + 0.55],
    ylim: [yLo, yHi],
    xlabel: "",
    ylabel: axisYlabelOf(SPEC?.axes, "main", "层内序号"),
    title: axisTitleOf(SPEC?.axes, "main", "依赖图（按最长路径分层）"),
    xticks: Array.from({ length: Math.floor(maxDepth) + 1 }, (_, i) => i),
    yticks: [],
    grid: false,
    autoscaled: false,
  });
  const maxErr = Math.max(1, ...Array.from(errorValues));
  const error = makeAxis("error", auxRect, {
    xlim: [-0.6, ERROR_LABELS.length - 0.4],
    ylim: [0, maxErr * 1.18],
    xlabel: "",
    ylabel: axisYlabelOf(SPEC?.axes, "error", "计数"),
    title: axisTitleOf(SPEC?.axes, "error", "依赖图的错误计数（应为 0）"),
    xticks: [],
    yticks: Array.from({ length: Math.floor(maxErr) + 1 }, (_, i) => i),
    grid: false,
    autoscaled: false,
  });
  scene.axes.push(main, error);

  // 边：先画全部（淡），再画高亮（实）。
  const baseSeries: LineSeries[] = [];
  const hlSeries: LineSeries[] = [];
  for (let i = 0; i < edgeX0.length; i += 1) {
    const seg = series([edgeX0[i] as number, edgeX1[i] as number], [edgeY0[i] as number, edgeY1[i] as number], pal.muted, {
      width: 0.8,
      alpha: 0.18,
    });
    if ((edgeHighlight[i] as number) > 0.5) {
      hlSeries.push(series([edgeX0[i] as number, edgeX1[i] as number], [edgeY0[i] as number, edgeY1[i] as number], pal.curves[0] as string, {
        width: 1.5,
        alpha: 0.9,
      }));
    } else {
      baseSeries.push(seg);
    }
  }
  if (baseSeries.length > 0) scene.layers.push(lineLayer("main", baseSeries));
  if (hlSeries.length > 0) scene.layers.push(lineLayer("main", hlSeries));

  // 节点：底层全体 + 焦点红圈 + 高亮彩色点。
  scene.layers.push(scatterLayer("main", nodeX, nodeY, { r: 4.2, color: pal.muted, alpha: 0.35 }));
  const focusX: number[] = [];
  const focusY: number[] = [];
  for (let i = 0; i < n; i += 1) {
    if ((focusMask[i] as number) > 0.5) {
      focusX.push(nodeX[i] as number);
      focusY.push(nodeY[i] as number);
    }
  }
  if (focusX.length > 0) {
    scene.layers.push(scatterLayer("main", focusX, focusY, { r: 9, color: pal.marker, alpha: 1 }));
  }
  const colors = partColors(sharedPalette(opts.theme));
  const groups = new Map<number, { x: number[]; y: number[] }>();
  for (let i = 0; i < n; i += 1) {
    if ((highlight[i] as number) <= 0.5) continue;
    const code = Math.round(nodePart[i] as number);
    let hit = groups.get(code);
    if (!hit) {
      hit = { x: [], y: [] };
      groups.set(code, hit);
    }
    hit.x.push(nodeX[i] as number);
    hit.y.push(nodeY[i] as number);
  }
  for (const [code, pts] of groups) {
    scene.layers.push(
      scatterLayer("main", pts.x, pts.y, { r: 5, color: colors[code] ?? pal.curves[0] as string, alpha: 0.95 }),
    );
  }

  // 节点标签：默认只标高亮闭包；选"全部"且不筛分部时才标全图（否则 28 个标签互相压）。
  const labelAll = (focus === "all" && part === "all") || n <= 10;
  for (let i = 0; i < n; i += 1) {
    const hot = (highlight[i] as number) > 0.5;
    if (!labelAll && !hot) continue;
    const [px, py] = dataToPixel(main, nodeX[i] as number, nodeY[i] as number);
    if (!Number.isFinite(px) || !Number.isFinite(py)) continue;
    scene.texts.push(
      text(px + 7, py - 6, ids[i] as string, {
        role: "label",
        size: 11,
        color: hot ? pal.ink : pal.muted,
        align: "left",
      }),
    );
  }

  // 副图：四格错误计数（0 也要有可见的数字标签）。
  scene.layers.push(
    barsLayer("error", ERROR_LABELS.map((_, i) => i), errorValues, {
      width: 0.55,
      color: nProblems > 0 ? pal.warn : pal.curves[1] as string,
      baseline: 0,
      label: "错误计数",
    }),
  );
  for (let i = 0; i < ERROR_LABELS.length; i += 1) {
    const v = errorValues[i] as number;
    const [px, py] = dataToPixel(error, i, v);
    if (Number.isFinite(px) && Number.isFinite(py)) {
      scene.texts.push(text(px, Math.max(error.rect[1] + 2, py - 15), fmt(v, 0), {
        role: "note",
        size: 11,
        color: v > 0 ? pal.warn : pal.muted,
        align: "center",
      }));
    }
    const [lx] = dataToPixel(error, i, 0);
    scene.texts.push(
      text(lx, error.rect[1] + error.rect[3] + 4, ERROR_LABELS[i] as string, {
        role: "axis",
        size: 11,
        color: pal.muted,
        align: "center",
      }),
    );
  }

  if (sidebar) {
    const depthCounts = new Map<number, number>();
    for (let i = 0; i < n; i += 1) depthCounts.set(nodeX[i] as number, (depthCounts.get(nodeX[i] as number) ?? 0) + 1);
    const widest = Math.max(...depthCounts.values());
    const chainNodes = chain.split("→").map((v) => v.trim()).filter((v) => v.length > 0);
    const chainLines: string[] = [];
    for (let i = 0; i < chainNodes.length; i += 3) {
      chainLines.push(chainNodes.slice(i, i + 3).join(" → "));
    }
    pushGroups(scene, sidebar.x, mainRect[1], auxRect[1] + auxRect[3], sidebar.w, [
      {
        lines: [
          `${fmt(n, 0)} 章、${fmt(nEdges, 0)} 条边、${fmt(metaNumber(frame, "n_layers"), 0)} 层。`,
          `最长关键路径 ${fmt(metaNumber(frame, "chain_len"), 0)} 章：`,
          ...chainLines,
          `最宽一层 ${widest} 章；层内序号只用于排布。`,
        ],
        color: pal.ink,
      },
      {
        lines: [closureNote],
        color: pal.muted,
      },
      {
        lines: [
          `焦点 ${focus}；分部筛选 ${part}。`,
          "节点颜色=分部；红圈=焦点章。",
          `问题：${problemNote}`,
        ],
        color: nProblems > 0 ? pal.warn : pal.muted,
      },
    ]);
  }

  scene.badges.push(placeBadge(sourceBadge(dataSourceOf(frame)), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge(
        "critical",
        `焦点 ${focus}：闭包 ${fmt(metaNumber(frame, "focus_closure"), 0)} 章 / ${fmt(metaNumber(frame, "focus_edges"), 0)} 条边`,
        8,
        44,
      ),
      SLOT.critical,
      width,
    ),
  );
  scene.badges.push(
    placeBadge(
      badge(
        "status",
        statusLine([
          `${fmt(n, 0)} 章`,
          `${fmt(nEdges, 0)} 边`,
          `${fmt(metaNumber(frame, "n_layers"), 0)} 层`,
          `问题 ${fmt(metaNumber(frame, "n_violations"), 0)}`,
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

