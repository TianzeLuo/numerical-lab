/**
 * 3.1 章的前端装配入口：可视化 id → model 的映射。
 *
 * 这一层刻意**不认识 DOM**（`view.ts` / `*.viz.ts` 才碰 DOM/canvas）：它只回答
 * 「这个 viz 用哪个 model 把帧变成 Scene」。`modelFor` 对未知 id 返回 `null` 而不是抛错——
 * 目录里可能有还没实现的 viz，调度层应当自己决定是提示还是跳过。
 *
 * 四个 viz 的分工：`order_convergence` 看**阶**（真跑的解 + 四条误差-步长曲线）、
 * `adaptive_step` 看**步长**（每次试步的 h、归一化 ε 与真误差）、
 * `order_conditions` 看**条件**（每棵有根树一条方程）、
 * `work_precision` 看**代价**（同一精度的反查工作量，无时间轴）。
 */

import type { SceneModel } from "../../shared/src/model.js";
import { adaptiveStepModel } from "./adaptive_step.model.js";
import { orderConditionsModel } from "./order_conditions.model.js";
import { orderConvergenceModel } from "./order_convergence.model.js";
import { workPrecisionModel } from "./work_precision.model.js";

/** viz id → model（与 `manifest.gen.ts` 的 `vizzes[].id` 一一对应）。 */
export const models: Record<string, SceneModel> = {
  order_convergence: orderConvergenceModel,
  adaptive_step: adaptiveStepModel,
  order_conditions: orderConditionsModel,
  work_precision: workPrecisionModel,
};

export { manifest } from "./manifest.gen.js";

/** 取某个 viz 的 model；未实现时返回 `null`（不抛错，调度层自行决定如何提示）。 */
export function modelFor(vizId: string): SceneModel | null {
  return Object.prototype.hasOwnProperty.call(models, vizId) ? models[vizId] : null;
}
