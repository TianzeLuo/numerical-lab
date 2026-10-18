/**
 * `order_convergence` 的挂载入口（章节可视化 → 页面）。理由同 `viz/chapters/1.2/lu_heatmap.viz.ts`：
 * 这一层才碰 DOM，`*.model.ts` 是纯函数（`Frame → Scene`），视觉回归正是走 model 那条路。
 *
 * 时间轴由 manifest 声明（`time.mode = "scrub"`）：`view.ts` 负责画进度条并把 `opts.t`
 * 传给 model；model 只读帧，滑块位置与真实积分时间的区别写在面板文字里（pitfalls 第 15 条）。
 */

import type { VizView, VizViewOptions } from "../../shared/src/view.js";
import { mountViz } from "../../shared/src/view.js";
import { manifest } from "./manifest.gen.js";
import { orderConvergenceModel } from "./order_convergence.model.js";

export const vizId = "order_convergence";

export function vizSpec() {
  const spec = manifest.vizzes.find((v) => v.id === vizId);
  if (!spec) throw new Error(`manifest 里没有 ${vizId}（manifest.gen.ts 需要重新生成？）`);
  return spec;
}

export function mount(host: HTMLElement, opts: Partial<VizViewOptions> = {}): Promise<VizView> {
  return mountViz({ manifest, viz: vizSpec(), model: orderConvergenceModel, host, ...opts });
}
