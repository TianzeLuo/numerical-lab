/**
 * `signal_spectrum` 的挂载入口（章节可视化 → 页面）。理由同 1.2 章的 `lu_heatmap.viz.ts`：
 * model 是纯函数（Frame→Scene），这一层只负责把 manifest spec、model 与宿主元素交给 `mountViz`。
 *
 * 时频双联动：时域波形（左列上）、单边 dB 谱（右列整高）与 |FFT−DFT| 对拍副轴（左列下）。
 * 时间轴 `mode="none"`：这一页没有可拖的时间，`t` 只是帧的索引（后端恒为 0）。
 */

import type { VizView, VizViewOptions } from "../../shared/src/view.js";
import { mountViz } from "../../shared/src/view.js";
import { manifest } from "./manifest.gen.js";
import { signalSpectrumModel } from "./signal_spectrum.model.js";

export const vizId = "signal_spectrum";

export function vizSpec() {
  const spec = manifest.vizzes.find((v) => v.id === vizId);
  if (!spec) throw new Error(`manifest 里没有 ${vizId}（manifest.gen.ts 需要重新生成？）`);
  return spec;
}

export function mount(host: HTMLElement, opts: Partial<VizViewOptions> = {}): Promise<VizView> {
  return mountViz({ manifest, viz: vizSpec(), model: signalSpectrumModel, host, ...opts });
}
