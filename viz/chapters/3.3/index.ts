/**
 * 3.3 章「几何数值积分与长期行为」的前端登记表：viz id → model。
 *
 * 只列**已经实现**的 viz；`modelFor` 对未实现的返回 `null`（不抛错），
 * 站点据此把没写好的那一页标成“未实现”而不是整章加载失败。
 */

import type { SceneModel } from "../../shared/src/model.js";
import { energyDriftDuelModel } from "./energy_drift_duel.model.js";
import { longTermOrbitModel } from "./long_term_orbit.model.js";
import { modifiedHamiltonianModel } from "./modified_hamiltonian.model.js";
import { phaseOrbitModel } from "./phase_orbit.model.js";

export const models: Record<string, SceneModel> = {
  energy_drift_duel: energyDriftDuelModel,
  phase_orbit: phaseOrbitModel,
  modified_hamiltonian: modifiedHamiltonianModel,
  long_term_orbit: longTermOrbitModel,
};

export { manifest } from "./manifest.gen.js";

export function modelFor(vizId: string): SceneModel | null {
  return Object.prototype.hasOwnProperty.call(models, vizId) ? models[vizId] : null;
}
