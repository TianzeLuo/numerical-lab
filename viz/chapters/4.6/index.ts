/**
 * 4.6 章「谱方法与伪谱」的前端登记表：viz id → model。
 *
 * 只列**已经实现**的 viz；`modelFor` 对未实现的返回 `null`（不抛错），
 * 站点据此把没写好的那一页标成「未实现」而不是整章加载失败。
 */

import type { SceneModel } from "../../shared/src/model.js";
import { burgersCascadeModel } from "./burgers_cascade.model.js";
import { convergenceRaceModel } from "./convergence_race.model.js";
import { dealiasRuleModel } from "./dealias_rule.model.js";
import { modalDecayModel } from "./modal_decay.model.js";

export const models: Record<string, SceneModel> = {
  modal_decay: modalDecayModel,
  dealias_rule: dealiasRuleModel,
  convergence_race: convergenceRaceModel,
  burgers_cascade: burgersCascadeModel,
};

export { manifest } from "./manifest.gen.js";
export { modalDecayModel } from "./modal_decay.model.js";
export { dealiasRuleModel } from "./dealias_rule.model.js";
export { convergenceRaceModel } from "./convergence_race.model.js";
export { burgersCascadeModel } from "./burgers_cascade.model.js";

export function modelFor(vizId: string): SceneModel | null {
  return Object.prototype.hasOwnProperty.call(models, vizId) ? models[vizId] : null;
}
