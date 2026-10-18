/**
 * `heat_ftcs` 的挂载入口（章节可视化 → 页面）。理由同 `oscillator_regimes.viz.ts`。
 *
 * 这个可视化额外需要 `history`（时空热力图）：`view.ts` 会把最近的帧攒起来传进 model，
 * 所以"拖动时间轴看热力图长出来"和"视觉回归里的单帧"用的是同一条代码路径。
 */

import type { VizView, VizViewOptions } from "../../shared/src/view.js";
import { mountViz } from "../../shared/src/view.js";
import { manifest } from "./manifest.gen.js";
import { heatFtcsModel } from "./heat_ftcs.model.js";

export const vizId = "heat_ftcs";

export function vizSpec() {
  const spec = manifest.vizzes.find((v) => v.id === vizId);
  if (!spec) throw new Error(`manifest 里没有 ${vizId}（manifest.gen.ts 需要重新生成？）`);
  return spec;
}

export function mount(host: HTMLElement, opts: Partial<VizViewOptions> = {}): Promise<VizView> {
  return mountViz({ manifest, viz: vizSpec(), model: heatFtcsModel, host, ...opts });
}
