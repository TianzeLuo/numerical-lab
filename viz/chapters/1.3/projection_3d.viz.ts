/**
 * `projection_3d` 的挂载入口（章节可视化 → 页面）。理由同 1.2 章 `lu_heatmap.viz.ts`。
 *
 * 时间轴是**分解动画进度**（manifest `time.mode = "scrub"`，`tMax = 1`）：`view.ts` 按
 * manifest 的 `time` 配置渲染进度条并把 `opts.t` 交给模型；模型只把 `t` 当"b = p_t + r_t
 * 拆到哪一步"用（`t=0` 只看得见 b，`t=1` 两个分量完全分开），真正的实测值全部来自帧。
 */

import type { VizView, VizViewOptions } from "../../shared/src/view.js";
import { mountViz } from "../../shared/src/view.js";
import { manifest } from "./manifest.gen.js";
import { projection3dModel } from "./projection_3d.model.js";

export const vizId = "projection_3d";

export function vizSpec() {
  const spec = manifest.vizzes.find((v) => v.id === vizId);
  if (!spec) throw new Error(`manifest 里没有 ${vizId}（manifest.gen.ts 需要重新生成？）`);
  return spec;
}

export function mount(host: HTMLElement, opts: Partial<VizViewOptions> = {}): Promise<VizView> {
  return mountViz({ manifest, viz: vizSpec(), model: projection3dModel, host, ...opts });
}
