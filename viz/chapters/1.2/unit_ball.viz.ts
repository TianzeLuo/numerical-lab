/**
 * `unit_ball` 的挂载入口（章节可视化 → 页面）。理由同 `lu_heatmap.viz.ts`。
 *
 * 这个可视化的主轴是**固定** ±3 的正方形面板（单位圆必须看起来是圆的），
 * 时间轴是 log10 κ：`view.ts` 的时间轴按 manifest 的 `time` 配置拖动，模型只把 t 当"当前 κ"用。
 */

import type { VizView, VizViewOptions } from "../../shared/src/view.js";
import { mountViz } from "../../shared/src/view.js";
import { manifest } from "./manifest.gen.js";
import { unitBallModel } from "./unit_ball.model.js";

export const vizId = "unit_ball";

export function vizSpec() {
  const spec = manifest.vizzes.find((v) => v.id === vizId);
  if (!spec) throw new Error(`manifest 里没有 ${vizId}（manifest.gen.ts 需要重新生成？）`);
  return spec;
}

export function mount(host: HTMLElement, opts: Partial<VizViewOptions> = {}): Promise<VizView> {
  return mountViz({ manifest, viz: vizSpec(), model: unitBallModel, host, ...opts });
}
