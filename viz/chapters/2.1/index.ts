/**
 * 2.1 章的前端装配入口：可视化 id → model 的映射。
 *
 * 这一层刻意**不认识 DOM**（`view.ts` / `*.viz.ts` 才碰 DOM/canvas）：它只回答
 * 「这个 viz 用哪个 model 把帧变成 Scene」。`modelFor` 对未知 id 返回 `null`
 * 而不是抛错——目录里可能有还没实现的 viz，调度层应当自己决定是提示还是跳过。
 *
 * 四个 viz 的分工：`runge_nodes` 看**节点分布**（同一函数、同一 n，只换分布）、
 * `lebesgue_growth` 看**常数**（发散被量化成 Λ_n 的指数率与它的斜率）、
 * `error_curve` 看**连续变形**（等距 → Chebyshev 的 t∈[0,1]）、
 * `barycentric_vs_vandermonde` 看**误差通道**（函数值 ×2 与系数 ×1 互不相干）。
 */

import type { SceneModel } from "../../shared/src/model.js";
import { barycentricVsVandermondeModel } from "./barycentric_vs_vandermonde.model.js";
import { errorCurveModel } from "./error_curve.model.js";
import { lebesgueGrowthModel } from "./lebesgue_growth.model.js";
import { rungeNodesModel } from "./runge_nodes.model.js";

/** viz id → model（与 `manifest.gen.ts` 的 `vizzes[].id` 一一对应）。 */
export const models: Record<string, SceneModel> = {
  runge_nodes: rungeNodesModel,
  lebesgue_growth: lebesgueGrowthModel,
  error_curve: errorCurveModel,
  barycentric_vs_vandermonde: barycentricVsVandermondeModel,
};

export { manifest } from "./manifest.gen.js";

/** 取某个 viz 的 model；未实现时返回 `null`（不抛错，调度层自行决定如何提示）。 */
export function modelFor(vizId: string): SceneModel | null {
  return Object.prototype.hasOwnProperty.call(models, vizId) ? models[vizId] : null;
}
