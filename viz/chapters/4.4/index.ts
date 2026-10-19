/**
 * 4.4 章「有限元方法」的前端登记表：viz id → model。
 *
 * 只列**已经实现**的 viz；`modelFor` 对未实现的返回 `null`（不抛错），
 * 站点据此把没写好的那一页标成「未实现」而不是整章加载失败。
 */

import type { SceneModel } from "../../shared/src/model.js";
import { adaptiveMeshModel } from "./adaptive_mesh.model.js";
import { assemblyAnimationModel } from "./assembly_animation.model.js";
import { hpConvergenceModel } from "./hp_convergence.model.js";
import { solveDeformModel } from "./solve_deform.model.js";

export const models: Record<string, SceneModel> = {
  assembly_animation: assemblyAnimationModel,
  solve_deform: solveDeformModel,
  hp_convergence: hpConvergenceModel,
  adaptive_mesh: adaptiveMeshModel,
};

export { manifest } from "./manifest.gen.js";
export { assemblyAnimationModel } from "./assembly_animation.model.js";
export { solveDeformModel } from "./solve_deform.model.js";
export { hpConvergenceModel } from "./hp_convergence.model.js";
export { adaptiveMeshModel } from "./adaptive_mesh.model.js";

export function modelFor(vizId: string): SceneModel | null {
  return Object.prototype.hasOwnProperty.call(models, vizId) ? models[vizId] : null;
}
