/**
 * `spline_vs_poly` 的挂载入口（章节可视化 → 页面）。
 *
 * 这一层薄到只剩"把 manifest 里的 spec、model、宿主元素交给 `mountViz`"：
 * 真正干活的在 `shared/src/view.ts`（DOM 装配）与 `spline_vs_poly.model.ts`（Frame→Scene）。
 * 站点按 `manifest.vizzes[].id` 找到这里，视觉回归则直接用 model——两条路共用同一个 model，
 * 所以"画出来的"和"回归测的"不可能不是同一张图。
 */

import type { VizView, VizViewOptions } from "../../shared/src/view.js";
import { mountViz } from "../../shared/src/view.js";
import { splineVsPolyModel } from "./spline_vs_poly.model.js";
import { manifest } from "./manifest.gen.js";

export const vizId = "spline_vs_poly";

export function vizSpec() {
  const spec = manifest.vizzes.find((v) => v.id === vizId);
  if (!spec) throw new Error(`manifest 里没有 ${vizId}（manifest.gen.ts 需要重新生成？）`);
  return spec;
}

export function mount(host: HTMLElement, opts: Partial<VizViewOptions> = {}): Promise<VizView> {
  return mountViz({ manifest, viz: vizSpec(), model: splineVsPolyModel, host, ...opts });
}
