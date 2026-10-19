/**
 * 3.4 章「边值问题与谱方法」的前端登记表：viz id → model。
 *
 * 只列**已经实现**的 viz；`modelFor` 对未实现的返回 `null`（不抛错），
 * 站点据此把没写好的那一页标成"未实现"而不是整章加载失败。
 */

import type { SceneModel } from "../../shared/src/model.js";
import { convergenceRateModel } from "./convergence_rate.model.js";
import { diffMatrixStructureModel } from "./diff_matrix_structure.model.js";
import { nodeClusteringModel } from "./node_clustering.model.js";
import { shootingScanModel } from "./shooting_scan.model.js";

export const models: Record<string, SceneModel> = {
  shooting_scan: shootingScanModel,
  convergence_rate: convergenceRateModel,
  node_clustering: nodeClusteringModel,
  diff_matrix_structure: diffMatrixStructureModel,
};

export { manifest } from "./manifest.gen.js";

export function modelFor(vizId: string): SceneModel | null {
  return Object.prototype.hasOwnProperty.call(models, vizId) ? models[vizId] : null;
}
