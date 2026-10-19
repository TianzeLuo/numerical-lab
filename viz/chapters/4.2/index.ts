/**
 * 4.2 章的 viz 登记表：viz id → 纯函数 model（不认识 DOM）。
 *
 * 四个 viz 都只消费 `server/chapters/4.2.py` 的帧字段；`manifest.gen.ts` 是契约，
 * 这里不重复任何参数或文案。
 */

import type { SceneModel } from "../../shared/src/model.js";
import { limiterInteractionModel } from "./limiter_interaction.model.js";
import { packetLongTimeModel } from "./packet_long_time.model.js";
import { phaseErrorModel } from "./phase_error.model.js";
import { schemeZooModel } from "./scheme_zoo.model.js";

export const models: Record<string, SceneModel> = {
  scheme_zoo: schemeZooModel,
  phase_error: phaseErrorModel,
  limiter_interaction: limiterInteractionModel,
  packet_long_time: packetLongTimeModel,
};

export { manifest } from "./manifest.gen.js";

export function modelFor(vizId: string): SceneModel | null {
  return Object.prototype.hasOwnProperty.call(models, vizId) ? models[vizId] : null;
}
