/**
 * 4.1 章的前端装配入口：可视化 id → model 的映射。
 *
 * 这一层刻意**不认识 DOM**（`view.ts` / `*.viz.ts` 才碰 DOM/canvas）：它只回答
 * 「这个 viz 用哪个 model 把帧变成 Scene」。`modelFor` 对未知 id 返回 `null`
 * 而不是抛错——目录里可能有还没实现的 viz，调度层应当自己决定是提示还是跳过。
 *
 * 四个 viz 的分工是"同一段数学，换一个地方看它会不会骗你"：
 * `stability_threshold` 看**判据**（|G| 越界而主图可能毫无异常）、
 * `convergence_order` 看**靶子**（选错靶子会得到相反的阶）、
 * `numerical_dissipation` 看**掩码**（分母/分子都可能是舍入级）、
 * `boundary_conservation` 看**结构性**（守恒来自行和为零，不是"数很小"）。
 */

import type { SceneModel } from "../../shared/src/model.js";
import { boundaryConservationModel } from "./boundary_conservation.model.js";
import { convergenceOrderModel } from "./convergence_order.model.js";
import { numericalDissipationModel } from "./numerical_dissipation.model.js";
import { stabilityThresholdModel } from "./stability_threshold.model.js";

/** viz id → model（与 `manifest.gen.ts` 的 `vizzes[].id` 一一对应）。 */
export const models: Record<string, SceneModel> = {
  stability_threshold: stabilityThresholdModel,
  convergence_order: convergenceOrderModel,
  numerical_dissipation: numericalDissipationModel,
  boundary_conservation: boundaryConservationModel,
};

export { manifest } from "./manifest.gen.js";

/** 取某个 viz 的 model；未实现时返回 `null`（不抛错，调度层自行决定如何提示）。 */
export function modelFor(vizId: string): SceneModel | null {
  return Object.prototype.hasOwnProperty.call(models, vizId) ? models[vizId] : null;
}
