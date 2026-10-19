/**
 * 5.1 章「无约束优化与拟牛顿」的前端登记表：viz id → model。
 * 只列已经实现的 viz；`modelFor` 对未实现的返回 null（站点标「未实现」而不是整章失败）。
 */

import type { SceneModel } from "../../shared/src/model.js";
import { descentRaceModel } from "./descent_race.model.js";
import { convergenceRatesModel } from "./convergence_rates.model.js";
import { saddleEscapeModel } from "./saddle_escape.model.js";
import { trustRegionRadiusModel } from "./trust_region_radius.model.js";

export const models: Record<string, SceneModel> = {
  descent_race: descentRaceModel,
  convergence_rates: convergenceRatesModel,
  saddle_escape: saddleEscapeModel,
  trust_region_radius: trustRegionRadiusModel,
};

export { manifest } from "./manifest.gen.js";
export { descentRaceModel } from "./descent_race.model.js";
export { convergenceRatesModel } from "./convergence_rates.model.js";
export { saddleEscapeModel } from "./saddle_escape.model.js";
export { trustRegionRadiusModel } from "./trust_region_radius.model.js";

export function modelFor(vizId: string): SceneModel | null {
  return Object.prototype.hasOwnProperty.call(models, vizId) ? models[vizId]! : null;
}
