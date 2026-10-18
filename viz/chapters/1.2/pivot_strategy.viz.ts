/**
 * `pivot_strategy` 的挂载入口（章节可视化 → 页面）。理由同 `lu_heatmap.viz.ts`。
 *
 * 这个可视化没有时间轴（manifest `time.mode = "none"`）：`view.ts` 不会渲染进度条，
 * 一切信息都在参数（矩阵、规模）里，模型收到的 `t` 恒为 0。
 */

import type { VizView, VizViewOptions } from "../../shared/src/view.js";
import { mountViz } from "../../shared/src/view.js";
import { manifest } from "./manifest.gen.js";
import { pivotStrategyModel } from "./pivot_strategy.model.js";

export const vizId = "pivot_strategy";

export function vizSpec() {
  const spec = manifest.vizzes.find((v) => v.id === vizId);
  if (!spec) throw new Error(`manifest 里没有 ${vizId}（manifest.gen.ts 需要重新生成？）`);
  return spec;
}

export function mount(host: HTMLElement, opts: Partial<VizViewOptions> = {}): Promise<VizView> {
  return mountViz({ manifest, viz: vizSpec(), model: pivotStrategyModel, host, ...opts });
}
