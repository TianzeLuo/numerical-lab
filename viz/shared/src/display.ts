/**
 * 网页显示层：把模型产出的完整 Scene 收敛成"只画图形"的版本。
 *
 * 模型把大量解释、读数、图例直接排进画布（它们要保证 PNG/视觉回归也能独立阅读）。
 * 网页有更好的排版能力：说明进 HTML 阅读区、图例进带色块的列表、长文进折叠面板。
 * 因此浏览器与视觉回归共用这一层变换：
 *
 *   - `texts` 清空、`badges` 保留但由 HTML 角标区呈现；画布只留图形；
 *   - 所有 `label` 收集成 HTML 图例；标签本身保留在图层里供命中测试与无障碍描述，
 *     两个渲染器按 `chrome.texts` 决定是否把它们画上画布；
 *   - 轴标题置空；短得下、放得开的 xlabel/ylabel 留在画布上，其余交给 HTML 面板说明；
 *   - 上下空白按坐标区与轴名裁剪，画布只留下图本身。
 *
 * 变换是纯函数：不碰帧、不重算数值，只搬运显示信息。
 */

import { cmapCssGradient } from "./colorscale.js";
import type { Axis, Badge, Layer, Scene, TextSpec } from "./scene.js";

export interface DisplayLegendItem {
  label: string;
  color: string;
  kind: Layer["kind"];
  /** 图例所属面板（同色重复时用来标 A/B）。 */
  axis: string;
  /** 热图图例用的 CSS 渐变；其它图层为 undefined。 */
  gradient?: string;
  /** 线型是否为虚线（同色多条曲线靠它区分）。 */
  dash?: boolean;
}

export interface DisplayAxisGuide {
  id: string;
  title: string;
  xlabel: string;
  ylabel: string;
  xscale: string;
  yscale: string;
}

export interface DisplaySceneResult {
  /** 交给 Canvas / 光栅器的干净场景。 */
  scene: Scene;
  /** 从图层 label 收集的图例（已去重）。 */
  legend: DisplayLegendItem[];
  /** 原始轴标题，供 HTML 阅读区逐面板说明。 */
  axes: DisplayAxisGuide[];
  /** 原始说明文字与角标：网页可以把它们放进折叠的"技术细节"。 */
  texts: TextSpec[];
  badges: Badge[];
}

function cleanLabel(label: string | undefined): string | undefined {
  if (label === undefined) return undefined;
  const text = label
    .replace(/\*\*|__|`/g, "")
    .replace(/[（(][^）)]{0,24}帧\s*meta\.[^）)]*[）)]/g, "")
    .replace(/\s+/g, " ")
    .trim();
  return text === "" ? undefined : text;
}

/**
 * 纵轴名竖排，能占用的高度就是面板高度。名字比面板还长时，它一定会伸进
 * 相邻面板或角标区——这类说明只有 HTML「面板」区能完整放下，所以画布上留空。
 *
 * 按光栅器的竖排字格（13px 字号下每个字符 17px）估算，两个渲染器取同一份场景，
 * 这样窄面板在基线与网页里都不会出现探出面板的纵轴名。
 */
function ylabelFits(label: string, axisHeight: number): boolean {
  return [...label].length * 17 <= Math.max(0, axisHeight - 10);
}

/** 横排文字宽度估算（CJK 按 12px、ASCII 按 6.2px，宁大勿小）。 */
function textWidthPx(label: string): number {
  let width = 0;
  for (const ch of label) {
    if (/\s/.test(ch)) width += 4;
    else if (/[\u2e80-\u9fff\uf900-\ufaff]/.test(ch)) width += 12;
    else if (/[A-Z0-9]/.test(ch)) width += 7.2;
    else width += 6.2;
  }
  return width;
}

/**
 * x 轴名横排。除了宽度，还要给它留出刻度数字下方约 30px 的带子；
 * 若正下方紧接着另一个面板，宁可把轴名放进 HTML 面板说明，也不压到下一张图上。
 */
function xlabelFits(axis: Axis, all: readonly Axis[], canvasHeight: number): boolean {
  const label = axis.xlabel.trim();
  if (label === "") return false;
  if (textWidthPx(label) > axis.rect[2] - 10) return false;
  const bottom = axis.rect[1] + axis.rect[3];
  const labelBottom = bottom + 31;
  if (labelBottom > canvasHeight - 2) return false;
  for (const other of all) {
    if (other === axis) continue;
    const top = other.rect[1];
    if (top > bottom + 2 && top < labelBottom + 4) return false;
  }
  return true;
}

/** 从面板标题里提取“横轴 = …”这类说明，作为没写 xlabel 时的回退。 */
function xlabelFromTitle(title: string): string {
  const match = /横轴\s*(?:=|是|为|：)\s*([^；;）)]+)/.exec(title);
  return match?.[1]?.trim() ?? "";
}

/**
 * 显示变换。`scene` 本身不被修改；返回的新 Scene 与调用方共享数据数组。
 *
 * 裁剪只动纵向：热力图的色标、右侧的标注都可能在坐标区之外，横向保持原宽更安全；
 * 上下则只有标题带/底部说明，`chrome` 已把它们关掉，裁掉是纯收益。
 */
export function sceneForDisplay(scene: Scene): DisplaySceneResult {
  const legend: DisplayLegendItem[] = [];
  const seen = new Set<string>();
  const addLegend = (
    label: string | undefined,
    color: string,
    kind: Layer["kind"],
    axis: string,
    extra: { gradient?: string; dash?: boolean } = {},
  ): void => {
    const text = cleanLabel(label);
    if (text === undefined) return;
    const key = `${kind}\u0000${color}\u0000${text}\u0000${extra.dash === true ? "dash" : "solid"}`;
    if (seen.has(key)) return;
    seen.add(key);
    const item: DisplayLegendItem = { label: text, color, kind, axis };
    if (extra.gradient !== undefined) item.gradient = extra.gradient;
    if (extra.dash === true) item.dash = true;
    legend.push(item);
  };

  const layers: Layer[] = scene.layers.map((layer) => {
    switch (layer.kind) {
      case "line": {
        const series = layer.series.map((s) => {
          const label = cleanLabel(s.label);
          addLegend(label, s.color, "line", layer.axes, { dash: s.dash !== undefined && s.dash.length > 0 });
          return { ...s, label };
        });
        return { ...layer, series };
      }
      case "band": {
        const label = cleanLabel(layer.band.label);
        addLegend(label, layer.band.color, "band", layer.axes);
        return { ...layer, band: { ...layer.band, label } };
      }
      case "scatter": {
        const label = cleanLabel(layer.label);
        addLegend(label, layer.color, "scatter", layer.axes);
        return { ...layer, label };
      }
      case "heat": {
        const label = cleanLabel(layer.label);
        addLegend(label, "#94a3b8", "heat", layer.axes, { gradient: cmapCssGradient(layer.cmap) });
        return { ...layer, label };
      }
      case "contour": {
        const label = cleanLabel(layer.label);
        addLegend(label, layer.color, "contour", layer.axes, { dash: true });
        return { ...layer, label };
      }
      case "bars": {
        const label = cleanLabel(layer.label);
        addLegend(label, layer.color, "bars", layer.axes);
        return { ...layer, label };
      }
      case "loglogfit": {
        const seriesLabel = cleanLabel(layer.seriesLabel) ?? layer.seriesLabel;
        const theoryLabel = cleanLabel(layer.theoryLabel) ?? layer.theoryLabel;
        addLegend(seriesLabel, layer.color ?? "#2b6cb0", "loglogfit", layer.axes);
        addLegend(theoryLabel, layer.theoryColor ?? "#c53030", "loglogfit", layer.axes, { dash: true });
        return { ...layer, seriesLabel, theoryLabel };
      }
      case "vectorfield":
        return layer;
    }
  });

  const xlabelKept = new Map<string, boolean>();
  for (const axis of scene.axes) xlabelKept.set(axis.id, xlabelFits(axis, scene.axes, scene.canvas.h));

  const axes: Axis[] = scene.axes.map((axis) => ({
    ...axis,
    rect: [axis.rect[0], axis.rect[1], axis.rect[2], axis.rect[3]],
    title: "",
    xlabel: xlabelKept.get(axis.id) === true ? axis.xlabel : "",
    ylabel: ylabelFits(axis.ylabel, axis.rect[3]) ? axis.ylabel : "",
  }));

  let height = scene.canvas.h;
  if (axes.length > 0) {
    const topEdge = Math.min(...axes.map((a) => a.rect[1]));
    const bottomEdge = Math.max(
      ...axes.map((a) => {
        const extra = a.xlabel !== "" ? 31 : a.xticks.length > 0 ? 18 : 8;
        return a.rect[1] + a.rect[3] + extra;
      }),
    );
    const top = Math.max(0, Math.floor(topEdge - 10));
    const bottom = Math.min(scene.canvas.h, Math.ceil(bottomEdge + 12));
    height = Math.max(80, bottom - top);
    for (const axis of axes) axis.rect[1] -= top;
  }

  const display: Scene = {
    ...scene,
    // 标题/徽标保留在 Scene 里（契约要求、也供 HTML 阅读区使用），
    // 但 chrome 关掉绘制：它们不属于图形本身。
    title: scene.title,
    canvas: { ...scene.canvas, h: height },
    axes,
    layers,
    texts: [],
    badges: scene.badges,
    chrome: { title: false, pitfalls: false, badges: false, texts: false },
  };

  return {
    scene: display,
    legend,
    axes: scene.axes.map((axis) => ({
      id: axis.id,
      title: axis.title,
      xlabel: axis.xlabel.trim() !== "" ? axis.xlabel : xlabelFromTitle(axis.title),
      ylabel: axis.ylabel,
      xscale: axis.xscale,
      yscale: axis.yscale,
    })),
    texts: scene.texts,
    badges: scene.badges,
  };
}
