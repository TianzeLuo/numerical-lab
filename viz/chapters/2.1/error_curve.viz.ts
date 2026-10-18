/**
 * `error_curve` 的挂载入口（章节可视化 → 页面）。样板与 `viz/chapters/1.2/lu_heatmap.viz.ts` 一致：
 * 这一层只做"manifest spec + model + 宿主元素 → mountViz"，不认识帧、也不认识 Scene。
 *
 * 误差曲线、包络与节点变形（时间轴 = 变形参数 t∈[0,1]，不是时间）
 */

import type { VizView, VizViewOptions } from "../../shared/src/view.js";
import { mountViz } from "../../shared/src/view.js";
import { manifest } from "./manifest.gen.js";
import { errorCurveModel } from "./error_curve.model.js";

export const vizId = "error_curve";

export function vizSpec() {
  const spec = manifest.vizzes.find((v) => v.id === vizId);
  if (!spec) throw new Error(`manifest 里没有 ${vizId}（manifest.gen.ts 需要重新生成？）`);
  return spec;
}

export function mount(host: HTMLElement, opts: Partial<VizViewOptions> = {}): Promise<VizView> {
  return mountViz({ manifest, viz: vizSpec(), model: errorCurveModel, host, ...opts });
}
