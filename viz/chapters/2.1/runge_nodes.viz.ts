/**
 * `runge_nodes` 的挂载入口（章节可视化 → 页面）。样板与 `viz/chapters/1.2/lu_heatmap.viz.ts` 一致：
 * 这一层只做"manifest spec + model + 宿主元素 → mountViz"，不认识帧、也不认识 Scene。
 *
 * 时间轴是「已用节点数 k」：拖到 k 就是在看"只用了前 k 个节点"的插值结果，
 * 于是 Runge 现象是**看**出来的（两端先炸、中间后炸），而不是从最大误差那一个数里读出来的。
 */

import type { VizView, VizViewOptions } from "../../shared/src/view.js";
import { mountViz } from "../../shared/src/view.js";
import { manifest } from "./manifest.gen.js";
import { rungeNodesModel } from "./runge_nodes.model.js";

export const vizId = "runge_nodes";

export function vizSpec() {
  const spec = manifest.vizzes.find((v) => v.id === vizId);
  if (!spec) throw new Error(`manifest 里没有 ${vizId}（manifest.gen.ts 需要重新生成？）`);
  return spec;
}

export function mount(host: HTMLElement, opts: Partial<VizViewOptions> = {}): Promise<VizView> {
  return mountViz({ manifest, viz: vizSpec(), model: rungeNodesModel, host, ...opts });
}
