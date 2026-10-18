/**
 * 4.5 章「多重网格与快速求解」的前端登记表：viz id → model。
 *
 * 只列**已经实现**的 viz；`modelFor` 对未实现的返回 `null`（不抛错），
 * 站点据此把没写好的那一页标成"未实现"而不是整章加载失败。
 */

import type { SceneModel } from "../../shared/src/model.js";
import { gridIndependenceModel } from "./grid_independence.model.js";
import { residualTransferModel } from "./residual_transfer.model.js";
import { smoothingSpectrumModel } from "./smoothing_spectrum.model.js";
import { vcycleHierarchyModel } from "./vcycle_hierarchy.model.js";

export const models: Record<string, SceneModel> = {
  smoothing_spectrum: smoothingSpectrumModel,
  vcycle_hierarchy: vcycleHierarchyModel,
  grid_independence: gridIndependenceModel,
  residual_transfer: residualTransferModel,
};

export { manifest } from "./manifest.gen.js";

export function modelFor(vizId: string): SceneModel | null {
  return Object.prototype.hasOwnProperty.call(models, vizId) ? models[vizId] : null;
}
