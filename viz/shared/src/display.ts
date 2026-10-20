/**
 * 网页显示层：把模型产出的完整 Scene 收敛成"只画图形"的版本。
 *
 * 模型把大量解释、读数、图例直接排进画布（它们要保证 PNG/视觉回归也能独立阅读）。
 * 网页有更好的排版能力：说明进 HTML 阅读区、图例进带色块的列表、长文进折叠面板。
 * 因此浏览器与视觉回归共用这一层变换：
 *
 *   - `texts` / `badges` 清空，`pitfalls` 保留但由 `chrome.pitfalls=false` 关掉绘制；
 *   - 所有 `label`（曲线/散点/热图/拟合标注）收集成 HTML 图例并从图层里移除；
 *   - 轴标题与 xlabel 置空（它们最容易在紧密堆叠的面板间互相压住），保留刻度与 ylabel；
 *   - 上下空白按坐标区裁剪，画布只留下图本身。
 *
 * 变换是纯函数：不碰帧、不重算数值，只搬运显示信息。
 */

import { cmapCssGradient } from "./colorscale.js";
import type { Axis, Badge, Layer, Scene, TextSpec } from "./scene.js";

export interface DisplayLegendItem {
  label: string;
  color: string;
  kind: Layer["kind"];
  /** 热图图例用的 CSS 渐变；其它图层为 undefined。 */
  gradient?: string;
}

export interface DisplayAxisGuide {
  id: string;
  title: string;
  ylabel: string;
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
  const text = label?.trim();
  return text === undefined || text === "" ? undefined : text;
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

/**
 * 显示变换。`scene` 本身不被修改；返回的新 Scene 与调用方共享数据数组。
 *
 * 裁剪只动纵向：热力图的色标、右侧的标注都可能在坐标区之外，横向保持原宽更安全；
 * 上下则只有标题带/底部说明，`chrome` 已把它们关掉，裁掉是纯收益。
 */
export function sceneForDisplay(scene: Scene): DisplaySceneResult {
  const legend: DisplayLegendItem[] = [];
  const seen = new Set<string>();
  const addLegend = (label: string | undefined, color: string, kind: Layer["kind"], gradient?: string): void => {
    const text = cleanLabel(label);
    if (text === undefined) return;
    const key = `${kind}\u0000${color}\u0000${text}`;
    if (seen.has(key)) return;
    seen.add(key);
    const item: DisplayLegendItem = { label: text, color, kind };
    if (gradient !== undefined) item.gradient = gradient;
    legend.push(item);
  };

  const layers: Layer[] = scene.layers.map((layer) => {
    switch (layer.kind) {
      case "line":
        return {
          ...layer,
          series: layer.series.map((s) => {
            addLegend(s.label, s.color, "line");
            const { label: _label, ...rest } = s;
            return rest;
          }),
        };
      case "band":
        addLegend(layer.band.label, layer.band.color, "band");
        return { ...layer, band: { ...layer.band, label: undefined } };
      case "scatter":
        addLegend(layer.label, layer.color, "scatter");
        return { ...layer, label: undefined };
      case "heat":
        addLegend(layer.label, "#94a3b8", "heat", cmapCssGradient(layer.cmap));
        return { ...layer, label: undefined };
      case "contour":
        addLegend(layer.label, layer.color, "contour");
        return { ...layer, label: undefined };
      case "bars":
        addLegend(layer.label, layer.color, "bars");
        return { ...layer, label: undefined };
      case "loglogfit": {
        const color = layer.color ?? "#2b6cb0";
        const theoryColor = layer.theoryColor ?? "#c53030";
        addLegend(layer.seriesLabel, color, "loglogfit");
        addLegend(layer.theoryLabel, theoryColor, "loglogfit");
        // 标注保留在图层里：契约要求两条线都有名字，渲染器据 chrome.texts 决定是否画上画布。
        return layer;
      }
      case "vectorfield":
        return layer;
    }
  });

  const axes: Axis[] = scene.axes.map((axis) => ({
    ...axis,
    rect: [axis.rect[0], axis.rect[1], axis.rect[2], axis.rect[3]],
    title: "",
    xlabel: "",
    ylabel: ylabelFits(axis.ylabel, axis.rect[3]) ? axis.ylabel : "",
  }));

  let height = scene.canvas.h;
  if (axes.length > 0) {
    const topEdge = Math.min(...axes.map((a) => a.rect[1]));
    const bottomEdge = Math.max(
      ...axes.map((a) => a.rect[1] + a.rect[3] + (a.xticks.length > 0 ? 18 : 8)),
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
      ylabel: axis.ylabel,
      yscale: axis.yscale,
    })),
    texts: scene.texts,
    badges: scene.badges,
  };
}
