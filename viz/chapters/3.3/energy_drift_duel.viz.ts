/** `energy_drift_duel` 的挂载入口：manifest spec + 纯函数 model + 宿主元素。 */

import type { VizView, VizViewOptions } from "../../shared/src/view.js";
import { mountViz } from "../../shared/src/view.js";
import { manifest } from "./manifest.gen.js";
import { energyDriftDuelModel } from "./energy_drift_duel.model.js";

export const vizId = "energy_drift_duel";

export function vizSpec() {
  const spec = manifest.vizzes.find((v) => v.id === vizId);
  if (!spec) throw new Error(`manifest 里没有 ${vizId}（manifest.gen.ts 需要重新生成？）`);
  return spec;
}

export function mount(host: HTMLElement, opts: Partial<VizViewOptions> = {}): Promise<VizView> {
  return mountViz({ manifest, viz: vizSpec(), model: energyDriftDuelModel, host, ...opts });
}
