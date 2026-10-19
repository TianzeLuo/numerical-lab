/**
 * 4.3 章「守恒律与有限体积」的前端登记表：viz id → model。
 *
 * 只列**已经实现**的 viz；`modelFor` 对未实现的返回 `null`（不抛错），
 * 站点据此把没写好的那一页标成"未实现"而不是整章加载失败。
 */

import type { SceneModel } from "../../shared/src/model.js";
import { burgersShockModel } from "./burgers_shock.model.js";
import { characteristicsModel } from "./characteristics.model.js";
import { conservationLedgerModel } from "./conservation_ledger.model.js";
import { riemannFanModel } from "./riemann_fan.model.js";

export const models: Record<string, SceneModel> = {
  burgers_shock: burgersShockModel,
  characteristics: characteristicsModel,
  conservation_ledger: conservationLedgerModel,
  riemann_fan: riemannFanModel,
};

export { manifest } from "./manifest.gen.js";

export function modelFor(vizId: string): SceneModel | null {
  return Object.prototype.hasOwnProperty.call(models, vizId) ? models[vizId] : null;
}
