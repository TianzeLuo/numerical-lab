/**
 * 6.1 章「二维综合仿真」的前端登记表：viz id → model。
 */

import type { SceneModel } from "../../shared/src/model.js";
import { cavityVorticityModel } from "./cavity_vorticity.model.js";
import { ghiaBenchmarkModel } from "./ghia_benchmark.model.js";
import { turingPatternModel } from "./turing_pattern.model.js";
import { damBreakModel } from "./dam_break.model.js";

export const models: Record<string, SceneModel> = {
  cavity_vorticity: cavityVorticityModel,
  ghia_benchmark: ghiaBenchmarkModel,
  turing_pattern: turingPatternModel,
  dam_break: damBreakModel,
};

export { manifest } from "./manifest.gen.js";
export { cavityVorticityModel } from "./cavity_vorticity.model.js";
export { ghiaBenchmarkModel } from "./ghia_benchmark.model.js";
export { turingPatternModel } from "./turing_pattern.model.js";
export { damBreakModel } from "./dam_break.model.js";

export function modelFor(vizId: string): SceneModel | null {
  return Object.prototype.hasOwnProperty.call(models, vizId) ? models[vizId]! : null;
}
