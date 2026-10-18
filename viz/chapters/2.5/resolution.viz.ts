/**
 * `resolution` 的挂载入口（章节可视化 → 页面）。理由同 1.2 章的 `lu_heatmap.viz.ts`：
 * model 是纯函数（Frame→Scene），这一层只负责把 manifest spec、model 与宿主元素交给 `mountViz`。
 *
 * 分辨率与不确定性：两音 dB 谱与分辨率尺（上方整宽）、能否分辨 vs 观测点数（左下）、|FFT−DFT| 对拍（右下）。
 * 时间轴 `mode="none"`；本页固定 T = 1，拖 N 只改变采样密度。
 */

import type { VizView, VizViewOptions } from "../../shared/src/view.js";
import { mountViz } from "../../shared/src/view.js";
import { manifest } from "./manifest.gen.js";
import { resolutionModel } from "./resolution.model.js";

export const vizId = "resolution";

export function vizSpec() {
  const spec = manifest.vizzes.find((v) => v.id === vizId);
  if (!spec) throw new Error(`manifest 里没有 ${vizId}（manifest.gen.ts 需要重新生成？）`);
  return spec;
}

export function mount(host: HTMLElement, opts: Partial<VizViewOptions> = {}): Promise<VizView> {
  return mountViz({ manifest, viz: vizSpec(), model: resolutionModel, host, ...opts });
}
