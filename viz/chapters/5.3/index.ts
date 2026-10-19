/**
 * 5.3 章「伴随方法与灵敏度」的前端登记表：viz id → model。
 */

import type { SceneModel } from "../../shared/src/model.js";
import { gradientFieldModel } from "./gradient_field.model.js";
import { influenceFieldModel } from "./influence_field.model.js";
import { gradientCheckModel } from "./gradient_check.model.js";
import { forwardAdjointPlaybackModel } from "./forward_adjoint_playback.model.js";

export const models: Record<string, SceneModel> = {
  gradient_field: gradientFieldModel,
  influence_field: influenceFieldModel,
  gradient_check: gradientCheckModel,
  forward_adjoint_playback: forwardAdjointPlaybackModel,
};

export { manifest } from "./manifest.gen.js";
export { gradientFieldModel } from "./gradient_field.model.js";
export { influenceFieldModel } from "./influence_field.model.js";
export { gradientCheckModel } from "./gradient_check.model.js";
export { forwardAdjointPlaybackModel } from "./forward_adjoint_playback.model.js";

export function modelFor(vizId: string): SceneModel | null {
  return Object.prototype.hasOwnProperty.call(models, vizId) ? models[vizId]! : null;
}
