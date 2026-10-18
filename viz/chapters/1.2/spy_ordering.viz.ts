/**
 * `spy_ordering` 的挂载入口（章节可视化 → 页面）。理由同 `lu_heatmap.viz.ts`。
 *
 * 时间轴是消元步 k：拖到哪一步，`fill_*` 就长到哪一步（`pending_*` 是"还没长出来的"），
 * 于是"消元即弦化"这件事是**看**出来的，而不是从文字里读出来的。
 */

import type { VizView, VizViewOptions } from "../../shared/src/view.js";
import { mountViz } from "../../shared/src/view.js";
import { manifest } from "./manifest.gen.js";
import { spyOrderingModel } from "./spy_ordering.model.js";

export const vizId = "spy_ordering";

export function vizSpec() {
  const spec = manifest.vizzes.find((v) => v.id === vizId);
  if (!spec) throw new Error(`manifest 里没有 ${vizId}（manifest.gen.ts 需要重新生成？）`);
  return spec;
}

export function mount(host: HTMLElement, opts: Partial<VizViewOptions> = {}): Promise<VizView> {
  return mountViz({ manifest, viz: vizSpec(), model: spyOrderingModel, host, ...opts });
}
