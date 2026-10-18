/**
 * `normal_vs_qr` 的挂载入口（章节可视化 → 页面）。理由同 1.2 章 `lu_heatmap.viz.ts`。
 *
 * 时间轴是**当前扫描位置**（manifest `time.mode = "scrub"`，`tMax = 14`，`label = "当前 log10 κ"`）：
 * `t` 就是 log10 κ，但真正被标注的实测值来自帧里的 `kappa_mark`/`error_mark`/`residual_mark`
 * ——拖到两个扫描点之间时，图上标的是最近的那个扫描点，不插值。
 */

import type { VizView, VizViewOptions } from "../../shared/src/view.js";
import { mountViz } from "../../shared/src/view.js";
import { manifest } from "./manifest.gen.js";
import { normalVsQrModel } from "./normal_vs_qr.model.js";

export const vizId = "normal_vs_qr";

export function vizSpec() {
  const spec = manifest.vizzes.find((v) => v.id === vizId);
  if (!spec) throw new Error(`manifest 里没有 ${vizId}（manifest.gen.ts 需要重新生成？）`);
  return spec;
}

export function mount(host: HTMLElement, opts: Partial<VizViewOptions> = {}): Promise<VizView> {
  return mountViz({ manifest, viz: vizSpec(), model: normalVsQrModel, host, ...opts });
}
