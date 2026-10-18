/**
 * `svd_truncation` 的挂载入口（章节可视化 → 页面）。理由同 1.2 章 `lu_heatmap.viz.ts`。
 *
 * 时间轴是**截断阶 k**（manifest `time.mode = "scrub"`，`tMax = 16`，`dt = 1`）：
 * 滑块连续，模型把 `t` 四舍五入到最近的整数 k 并夹在 `k_grid` 范围内，
 * 显示出来的当前 k 取自帧 `meta.k_current`（不做"t 就是 k"的假设）。
 */

import type { VizView, VizViewOptions } from "../../shared/src/view.js";
import { mountViz } from "../../shared/src/view.js";
import { manifest } from "./manifest.gen.js";
import { svdTruncationModel } from "./svd_truncation.model.js";

export const vizId = "svd_truncation";

export function vizSpec() {
  const spec = manifest.vizzes.find((v) => v.id === vizId);
  if (!spec) throw new Error(`manifest 里没有 ${vizId}（manifest.gen.ts 需要重新生成？）`);
  return spec;
}

export function mount(host: HTMLElement, opts: Partial<VizViewOptions> = {}): Promise<VizView> {
  return mountViz({ manifest, viz: vizSpec(), model: svdTruncationModel, host, ...opts });
}
