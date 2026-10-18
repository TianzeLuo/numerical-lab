/**
 * `uncertainty_fan` 的挂载入口（章节可视化 → 页面）。理由同 1.2 章 `lu_heatmap.viz.ts`。
 *
 * 时间轴是**噪声方向角 θ**（manifest `time.mode = "scrub"`，`tMax = 360`，`dt = 5`）：
 * 模型把 `t` 取模到 [0,360) 后找**最近的采样方向**（帧里的实测解），不做插值——
 * 插值出来的解不是任何一次实测的结果，画上去就成了编数据。
 */

import type { VizView, VizViewOptions } from "../../shared/src/view.js";
import { mountViz } from "../../shared/src/view.js";
import { manifest } from "./manifest.gen.js";
import { uncertaintyFanModel } from "./uncertainty_fan.model.js";

export const vizId = "uncertainty_fan";

export function vizSpec() {
  const spec = manifest.vizzes.find((v) => v.id === vizId);
  if (!spec) throw new Error(`manifest 里没有 ${vizId}（manifest.gen.ts 需要重新生成？）`);
  return spec;
}

export function mount(host: HTMLElement, opts: Partial<VizViewOptions> = {}): Promise<VizView> {
  return mountViz({ manifest, viz: vizSpec(), model: uncertaintyFanModel, host, ...opts });
}
