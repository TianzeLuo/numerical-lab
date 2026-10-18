/**
 * `conv_theorem` 的挂载入口（章节可视化 → 页面）。理由同 1.2 章的 `lu_heatmap.viz.ts`：
 * model 是纯函数（Frame→Scene），这一层只负责把 manifest spec、model 与宿主元素交给 `mountViz`。
 *
 * 卷积定理与卷绕：两条卷积曲线（上方整宽）、循环−线性的差（左下）、补零到 N+M−1 后的差（右下）。
 * 时间轴 `mode="none"`。卷绕误差量级 0.25 vs 补零后 1e-16，并排才读得出「差四个数量级」。
 */

import type { VizView, VizViewOptions } from "../../shared/src/view.js";
import { mountViz } from "../../shared/src/view.js";
import { manifest } from "./manifest.gen.js";
import { convTheoremModel } from "./conv_theorem.model.js";

export const vizId = "conv_theorem";

export function vizSpec() {
  const spec = manifest.vizzes.find((v) => v.id === vizId);
  if (!spec) throw new Error(`manifest 里没有 ${vizId}（manifest.gen.ts 需要重新生成？）`);
  return spec;
}

export function mount(host: HTMLElement, opts: Partial<VizViewOptions> = {}): Promise<VizView> {
  return mountViz({ manifest, viz: vizSpec(), model: convTheoremModel, host, ...opts });
}
