/**
 * `barycentric_vs_vandermonde` 的挂载入口（章节可视化 → 页面）。样板与 `viz/chapters/1.2/lu_heatmap.viz.ts` 一致：
 * 这一层只做"manifest spec + model + 宿主元素 → mountViz"，不认识帧、也不认识 Scene。
 *
 * 重心公式 vs 直接解 Vandermonde：三条误差通道（时间轴 = 节点数 n）
 */

import type { VizView, VizViewOptions } from "../../shared/src/view.js";
import { mountViz } from "../../shared/src/view.js";
import { manifest } from "./manifest.gen.js";
import { barycentricVsVandermondeModel } from "./barycentric_vs_vandermonde.model.js";

export const vizId = "barycentric_vs_vandermonde";

export function vizSpec() {
  const spec = manifest.vizzes.find((v) => v.id === vizId);
  if (!spec) throw new Error(`manifest 里没有 ${vizId}（manifest.gen.ts 需要重新生成？）`);
  return spec;
}

export function mount(host: HTMLElement, opts: Partial<VizViewOptions> = {}): Promise<VizView> {
  return mountViz({ manifest, viz: vizSpec(), model: barycentricVsVandermondeModel, host, ...opts });
}
