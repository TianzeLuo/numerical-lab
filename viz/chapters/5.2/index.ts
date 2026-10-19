/**
 * 5.2 章「反问题与正则化」的前端登记表：viz id → model。
 */

import type { SceneModel } from "../../shared/src/model.js";
import { lcurveTradeoffModel } from "./lcurve_tradeoff.model.js";
import { filterShapesModel } from "./filter_shapes.model.js";
import { imageDeconvolutionModel } from "./image_deconvolution.model.js";
import { biasVarianceModel } from "./bias_variance.model.js";

export const models: Record<string, SceneModel> = {
  lcurve_tradeoff: lcurveTradeoffModel,
  filter_shapes: filterShapesModel,
  image_deconvolution: imageDeconvolutionModel,
  bias_variance: biasVarianceModel,
};

export { manifest } from "./manifest.gen.js";
export { lcurveTradeoffModel } from "./lcurve_tradeoff.model.js";
export { filterShapesModel } from "./filter_shapes.model.js";
export { imageDeconvolutionModel } from "./image_deconvolution.model.js";
export { biasVarianceModel } from "./bias_variance.model.js";

export function modelFor(vizId: string): SceneModel | null {
  return Object.prototype.hasOwnProperty.call(models, vizId) ? models[vizId]! : null;
}
