/**
 * `phase_error` 的挂载入口：把 manifest spec、model、宿主元素交给 `mountViz`。
 *
 * 与 4.5 / 2.6 章的 `<viz>.viz.ts` 同一个样板：这一层**不认识帧**，
 * 只负责「哪一份 manifest + 哪一个纯函数模型」。
 */

import type { VizView, VizViewOptions } from "../../shared/src/view.js";
import { mountViz } from "../../shared/src/view.js";
import { manifest } from "./manifest.gen.js";
import { phaseErrorModel } from "./phase_error.model.js";

export const vizId = "phase_error";

export function vizSpec() {
  const spec = manifest.vizzes.find((v) => v.id === vizId);
  if (!spec) throw new Error(`manifest 里没有 ${vizId}（manifest.gen.ts 需要重新生成？）`);
  return spec;
}

export function mount(host: HTMLElement, opts: Partial<VizViewOptions> = {}): Promise<VizView> {
  return mountViz({ manifest, viz: vizSpec(), model: phaseErrorModel, host, ...opts });
}
