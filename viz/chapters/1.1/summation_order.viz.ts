/**
 * `summation_order` 的挂载入口（章节可视化 → 页面）。
 *
 * 这一层刻意只有样板代码：model 是纯函数（帧 → Scene），DOM/时间轴/canvas 都在
 * `shared/src/view.ts` 里。五种求和顺序与两条界 的**时间轴语义**见 model 文件头。
 */

import type { VizView, VizViewOptions } from "../../shared/src/view.js";
import { mountViz } from "../../shared/src/view.js";
import { manifest } from "./manifest.gen.js";
import { summationOrderModel } from "./summation_order.model.js";

export const vizId = "summation_order";

export function vizSpec() {
  const spec = manifest.vizzes.find((v) => v.id === vizId);
  if (!spec) throw new Error(`manifest 里没有 ${vizId}（manifest.gen.ts 需要重新生成？）`);
  return spec;
}

export function mount(host: HTMLElement, opts: Partial<VizViewOptions> = {}): Promise<VizView> {
  return mountViz({ manifest, viz: vizSpec(), model: summationOrderModel, host, ...opts });
}
