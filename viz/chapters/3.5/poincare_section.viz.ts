/**
 * `poincare_section` 的挂载入口：把 manifest spec、model 与宿主元素交给 `mountViz`
 * （样板与 `0.0/heat_ftcs.viz.ts` 逐行一致；这一层才碰 DOM/canvas）。
 */

import type { VizView, VizViewOptions } from "../../shared/src/view.js";
import { mountViz } from "../../shared/src/view.js";
import { manifest } from "./manifest.gen.js";
import { poincareSectionModel } from "./poincare_section.model.js";

export const vizId = "poincare_section";

export function vizSpec() {
  const spec = manifest.vizzes.find((v) => v.id === vizId);
  if (!spec) throw new Error(`manifest 里没有 ${vizId}（manifest.gen.ts 需要重新生成？）`);
  return spec;
}

export function mount(host: HTMLElement, opts: Partial<VizViewOptions> = {}): Promise<VizView> {
  return mountViz({ manifest, viz: vizSpec(), model: poincareSectionModel, host, ...opts });
}
