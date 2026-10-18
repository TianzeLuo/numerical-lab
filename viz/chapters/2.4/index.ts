/**
 * 2.4 章的前端装配入口：可视化 id → model 的映射。
 *
 * 这一层刻意**不认识 DOM**（`view.ts` / `*.viz.ts` 才碰 DOM/canvas）：它只回答
 * 「这个 viz 用哪个 model 把帧变成 Scene」。`modelFor` 对未知 id 返回 `null`
 * 而不是抛错——调度层应当自己决定是提示还是跳过。
 *
 * 四个 viz 的分工（对应笔记的四节）：
 *  - `area_convergence` 看**阶**：面板数 → 误差的幂律，以及"量到的是 f 光滑度的阶"；
 *  - `gauss_vs_equispaced` 看**精确度的两个方向**：2n−1 次精确 vs 权重全正（Γ≡1）；
 *  - `romberg_table` 看**外推**：逐列的阶、对角线的地板、外推的代价 A_j；
 *  - `singular_endpoint` 看**奇异性**：双指数变换的地板来自"没采到的尾巴"，
 *    而换元能把阶还回来（代价是必须先知道 α）。
 */

import type { SceneModel } from "../../shared/src/model.js";
import { areaConvergenceModel } from "./area_convergence.model.js";
import { gaussVsEquispacedModel } from "./gauss_vs_equispaced.model.js";
import { rombergTableModel } from "./romberg_table.model.js";
import { singularEndpointModel } from "./singular_endpoint.model.js";

/** viz id → model（与 `manifest.gen.ts` 的 `vizzes[].id` 一一对应）。 */
export const models: Record<string, SceneModel> = {
  area_convergence: areaConvergenceModel,
  gauss_vs_equispaced: gaussVsEquispacedModel,
  romberg_table: rombergTableModel,
  singular_endpoint: singularEndpointModel,
};

export { manifest } from "./manifest.gen.js";

/** 取某个 viz 的 model；未实现时返回 `null`（不抛错，调度层自行决定如何提示）。 */
export function modelFor(vizId: string): SceneModel | null {
  return Object.prototype.hasOwnProperty.call(models, vizId) ? models[vizId] : null;
}
