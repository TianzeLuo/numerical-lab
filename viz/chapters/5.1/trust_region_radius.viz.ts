/**
 * `trust_region_radius` 的挂载入口：manifest spec + 纯函数 model + 宿主元素 → `mountViz`。
 * 这一层不认识帧，只负责把三者拼起来。
 */

import type { VizView, VizViewOptions } from "../../shared/src/view.js";
import { mountViz } from "../../shared/src/view.js";
import { manifest } from "./manifest.gen.js";
import { trustRegionRadiusModel } from "./trust_region_radius.model.js";

export const vizId = "trust_region_radius";

export function vizSpec() {
  const spec = manifest.vizzes.find((v) => v.id === vizId);
  if (!spec) throw new Error(`manifest 里没有 ${vizId}（manifest.gen.ts 需要重新生成？）`);
  return spec;
}

export function mount(host: HTMLElement, opts: Partial<VizViewOptions> = {}): Promise<VizView> {
  return mountViz({ manifest, viz: vizSpec(), model: trustRegionRadiusModel, host, ...opts });
}
