/**
 * `ghia_benchmark` 的挂载入口：manifest spec + 纯函数 model + 宿主元素 → `mountViz`。
 */

import type { VizView, VizViewOptions } from "../../shared/src/view.js";
import { mountViz } from "../../shared/src/view.js";
import { manifest } from "./manifest.gen.js";
import { ghiaBenchmarkModel } from "./ghia_benchmark.model.js";

export const vizId = "ghia_benchmark";

export function vizSpec() {
  const spec = manifest.vizzes.find((v) => v.id === vizId);
  if (!spec) throw new Error(`manifest 里没有 ${vizId}（manifest.gen.ts 需要重新生成？）`);
  return spec;
}

export function mount(host: HTMLElement, opts: Partial<VizViewOptions> = {}): Promise<VizView> {
  return mountViz({ manifest, viz: vizSpec(), model: ghiaBenchmarkModel, host, ...opts });
}
