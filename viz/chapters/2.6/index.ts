/**
 * 2.6 章「小波与多分辨分析」的前端登记表：viz id → model。
 *
 * 只列**已经实现**的 viz；`modelFor` 对未实现的返回 `null`（不抛错），
 * 站点据此把没写好的那一页标成"未实现"而不是整章加载失败。
 */

import type { SceneModel } from "../../shared/src/model.js";
import { decompTreeModel } from "./decomp_tree.model.js";
import { denoiseThresholdModel } from "./denoise_threshold.model.js";
import { timeScaleMapModel } from "./time_scale_map.model.js";
import { vanishingMomentsModel } from "./vanishing_moments.model.js";

export const models: Record<string, SceneModel> = {
  decomp_tree: decompTreeModel,
  time_scale_map: timeScaleMapModel,
  denoise_threshold: denoiseThresholdModel,
  vanishing_moments: vanishingMomentsModel,
};

export { manifest } from "./manifest.gen.js";

export function modelFor(vizId: string): SceneModel | null {
  return Object.prototype.hasOwnProperty.call(models, vizId) ? models[vizId] : null;
}
