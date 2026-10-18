/**
 * 3.2 章的前端装配入口：可视化 id → model 的映射。
 *
 * 这一层刻意**不认识 DOM**（`view.ts` / `*.viz.ts` 才碰 DOM/canvas）：它只回答
 * 「这个 viz 用哪个 model 把帧变成 Scene」。`modelFor` 对未知 id 返回 `null`
 * 而不是抛错——目录里可能有还没实现的 viz，调度层应当自己决定是提示还是跳过。
 *
 * 五个 viz 的分工：`stability_region` 看**落点**（λh 在复平面上的位置与解的真实发散）、
 * `stiff_dilemma` 看**困境**（解平缓但步长被 λ_max 掐住）、`stability_surface` 看**曲面**
 * （|R(z)| 与精确边界；时间轴为 none）、`explicit_vs_implicit` 看**总账**（步长/误差/计算量）、
 * `bdf_angle` 看**角度**（A(α) 楔形随阶数收窄）。
 *
 * 五个 model 共用 `_panel.ts` 的排版与计数工具；**没有任何一个在本文件里重算物理量**：
 * 所有数字都来自帧数组或帧 meta/strings。
 */

import type { SceneModel } from "../../shared/src/model.js";
import { bdfAngleModel } from "./bdf_angle.model.js";
import { explicitVsImplicitModel } from "./explicit_vs_implicit.model.js";
import { stabilityRegionModel } from "./stability_region.model.js";
import { stabilitySurfaceModel } from "./stability_surface.model.js";
import { stiffDilemmaModel } from "./stiff_dilemma.model.js";

/** viz id → model（与 `manifest.gen.ts` 的 `vizzes[].id` 一一对应）。 */
export const models: Record<string, SceneModel> = {
  stability_region: stabilityRegionModel,
  stiff_dilemma: stiffDilemmaModel,
  stability_surface: stabilitySurfaceModel,
  explicit_vs_implicit: explicitVsImplicitModel,
  bdf_angle: bdfAngleModel,
};

export { manifest } from "./manifest.gen.js";

/** 取某个 viz 的 model；未实现时返回 `null`（不抛错，调度层自行决定如何提示）。 */
export function modelFor(vizId: string): SceneModel | null {
  return Object.prototype.hasOwnProperty.call(models, vizId) ? models[vizId] : null;
}
