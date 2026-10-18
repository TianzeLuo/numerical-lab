/**
 * 2.3 章的前端装配入口：可视化 id → model 的映射。
 *
 * 这一层刻意**不认识 DOM**（`view.ts` / `*.viz.ts` 才碰 DOM/canvas）：它只回答
 * 「这个 viz 用哪个 model 把帧变成 Scene」。`modelFor` 对未知 id 返回 `null`
 * 而不是抛错——目录里可能有还没实现的 viz，调度层应当自己决定是提示还是跳过。
 *
 * 四个 viz 的分工（对应笔记的四节）：
 *  - `projection` 看**几何**：L² 投影 = 正交投影，换基不改函数、只改系数与条件数；
 *  - `convergence_rates` 看**速率**：光滑性决定误差怎么衰减（实测斜率 vs 理论斜率）；
 *  - `equioscillation` 看**一致逼近**：Remez 的 n+2 个等幅交错点，以及它与 L² 的差别；
 *  - `zeros_evolution` 看**节点从哪来**：三族正交多项式的零点分布与求根精度。
 */

import type { SceneModel } from "../../shared/src/model.js";
import { convergenceRatesModel } from "./convergence_rates.model.js";
import { equioscillationModel } from "./equioscillation.model.js";
import { projectionModel } from "./projection.model.js";
import { zerosEvolutionModel } from "./zeros_evolution.model.js";

/** viz id → model（与 `manifest.gen.ts` 的 `vizzes[].id` 一一对应）。 */
export const models: Record<string, SceneModel> = {
  projection: projectionModel,
  convergence_rates: convergenceRatesModel,
  equioscillation: equioscillationModel,
  zeros_evolution: zerosEvolutionModel,
};

export { manifest } from "./manifest.gen.js";

/** 取某个 viz 的 model；未实现时返回 `null`（不抛错，调度层自行决定如何提示）。 */
export function modelFor(vizId: string): SceneModel | null {
  return Object.prototype.hasOwnProperty.call(models, vizId) ? models[vizId] : null;
}
