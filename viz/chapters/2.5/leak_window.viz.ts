/**
 * `leak_window` 的挂载入口（章节可视化 → 页面）。理由同 1.2 章的 `lu_heatmap.viz.ts`：
 * model 是纯函数（Frame→Scene），这一层只负责把 manifest spec、model 与宿主元素交给 `mountViz`。
 *
 * 泄漏与窗：四种窗形（左列上）、加窗后的 dB 谱与渐近包络（右列整高）、解析窗谱对拍（左列下）。
 * `offset` 是频率偏离栅格的格数（非整周期截断才会泄漏）；`pad` 只加密栅格。
 */

import type { VizView, VizViewOptions } from "../../shared/src/view.js";
import { mountViz } from "../../shared/src/view.js";
import { manifest } from "./manifest.gen.js";
import { leakWindowModel } from "./leak_window.model.js";

export const vizId = "leak_window";

export function vizSpec() {
  const spec = manifest.vizzes.find((v) => v.id === vizId);
  if (!spec) throw new Error(`manifest 里没有 ${vizId}（manifest.gen.ts 需要重新生成？）`);
  return spec;
}

export function mount(host: HTMLElement, opts: Partial<VizViewOptions> = {}): Promise<VizView> {
  return mountViz({ manifest, viz: vizSpec(), model: leakWindowModel, host, ...opts });
}
