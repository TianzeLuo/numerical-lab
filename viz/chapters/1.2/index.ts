/**
 * 1.2 章的前端装配入口：可视化 id → model 的映射。
 *
 * 这一层刻意**不认识 DOM**（`view.ts` / `*.viz.ts` 才碰 DOM/canvas）：它只回答
 * 「这个 viz 用哪个 model 把帧变成 Scene」。`modelFor` 对未知 id 返回 `null`
 * 而不是抛错——目录里可能有还没实现的 viz，调度层应当自己决定是提示还是跳过。
 *
 * 四个 viz 的分工：`lu_heatmap` 看**过程**（消元每一步的矩阵与 ρ/残差）、
 * `unit_ball` 看**几何**（单位圆的像、|det|、κ）、`spy_ordering` 看**结构**（填充怎么长出来）、
 * `pivot_strategy` 看**策略**（三种主元选择的 ρ 与前后向误差）。
 */

import type { SceneModel } from "../../shared/src/model.js";
import { luHeatmapModel } from "./lu_heatmap.model.js";
import { pivotStrategyModel } from "./pivot_strategy.model.js";
import { spyOrderingModel } from "./spy_ordering.model.js";
import { unitBallModel } from "./unit_ball.model.js";

/** viz id → model（与 `manifest.gen.ts` 的 `vizzes[].id` 一一对应）。 */
export const models: Record<string, SceneModel> = {
  lu_heatmap: luHeatmapModel,
  unit_ball: unitBallModel,
  spy_ordering: spyOrderingModel,
  pivot_strategy: pivotStrategyModel,
};

export { manifest } from "./manifest.gen.js";

/** 取某个 viz 的 model；未实现时返回 `null`（不抛错，调度层自行决定如何提示）。 */
export function modelFor(vizId: string): SceneModel | null {
  return Object.prototype.hasOwnProperty.call(models, vizId) ? models[vizId] : null;
}
