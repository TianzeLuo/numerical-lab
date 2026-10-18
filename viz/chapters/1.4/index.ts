/**
 * 1.4 章的前端装配入口：可视化 id → model 的映射。
 *
 * 这一层刻意**不认识 DOM**（`view.ts` / `*.viz.ts` 才碰 DOM/canvas）：它只回答
 * 「这个 viz 用哪个 model 把帧变成 Scene」。`modelFor` 对未知 id 返回 `null`
 * 而不是抛错——目录里可能有还没实现的 viz，调度层应当自己决定是提示还是跳过。
 *
 * 五个 viz 的分工：`power_iteration` 看**迭代向量怎么对齐主特征方向**、
 * `qr_schur` 看**次对角元怎么被逐个压掉**、`pseudospectrum` 看**谱之外还有多大一片**、
 * `transient_growth` 看**谱半径说会衰减而范数先涨十个量级**、
 * `gmres_pseudospec` 看**只依赖谱的界与实际残差怎么分道扬镳**。
 */

import type { SceneModel } from "../../shared/src/model.js";
import { gmresPseudospecModel } from "./gmres_pseudospec.model.js";
import { powerIterationModel } from "./power_iteration.model.js";
import { pseudospectrumModel } from "./pseudospectrum.model.js";
import { qrSchurModel } from "./qr_schur.model.js";
import { transientGrowthModel } from "./transient_growth.model.js";

/** viz id → model（与 `manifest.gen.ts` 的 `vizzes[].id` 一一对应）。 */
export const models: Record<string, SceneModel> = {
  power_iteration: powerIterationModel,
  qr_schur: qrSchurModel,
  pseudospectrum: pseudospectrumModel,
  transient_growth: transientGrowthModel,
  gmres_pseudospec: gmresPseudospecModel,
};

export { manifest } from "./manifest.gen.js";

/** 取某个 viz 的 model；未实现时返回 `null`（不抛错，调度层自行决定如何提示）。 */
export function modelFor(vizId: string): SceneModel | null {
  return Object.prototype.hasOwnProperty.call(models, vizId) ? models[vizId] : null;
}
