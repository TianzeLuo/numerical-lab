/**
 * 5.4 章「UQ 与全局敏感性」的前端登记表：viz id → model。
 */

import type { SceneModel } from "../../shared/src/model.js";
import { pointFillModel } from "./point_fill.model.js";
import { outputHistModel } from "./output_hist.model.js";
import { sobolBarsModel } from "./sobol_bars.model.js";
import { pceSpectrumModel } from "./pce_spectrum.model.js";

export const models: Record<string, SceneModel> = {
  point_fill: pointFillModel,
  output_hist: outputHistModel,
  sobol_bars: sobolBarsModel,
  pce_spectrum: pceSpectrumModel,
};

export { manifest } from "./manifest.gen.js";
export { pointFillModel } from "./point_fill.model.js";
export { outputHistModel } from "./output_hist.model.js";
export { sobolBarsModel } from "./sobol_bars.model.js";
export { pceSpectrumModel } from "./pce_spectrum.model.js";

export function modelFor(vizId: string): SceneModel | null {
  return Object.prototype.hasOwnProperty.call(models, vizId) ? models[vizId]! : null;
}
