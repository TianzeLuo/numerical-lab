/**
 * 1.5 章的前端装配入口：可视化 id → model 的映射。
 *
 * 这一层刻意**不认识 DOM**（`view.ts` / `*.viz.ts` 才碰 DOM/canvas）：它只回答
 * 「这个 viz 用哪个 model 把帧变成 Scene」。`modelFor` 对未知 id 返回 `null`
 * 而不是抛错——目录里可能有还没实现的 viz，调度层应当自己决定是提示还是跳过。
 *
 * 四个 viz 的分工是"从轨迹、到模态、到子空间、到预条件"：
 * `iteration_race` 看**轨迹**（五种定常/Krylov 方法在同一张椭圆等高线上赛跑）、
 * `error_spectrum` 看**模态**（谁先衰减完、实测比值与解析因子差多少）、
 * `krylov_growth` 看**子空间**（基逐列扩张 + 残差正交性的实测矩阵）、
 * `precond_effect` 看**预条件**（谱分布被压到哪里，以及 κ 与迭代数并不是一回事）。
 */

import type { SceneModel } from "../../shared/src/model.js";
import { errorSpectrumModel } from "./error_spectrum.model.js";
import { iterationRaceModel } from "./iteration_race.model.js";
import { krylovGrowthModel } from "./krylov_growth.model.js";
import { precondEffectModel } from "./precond_effect.model.js";

/** viz id → model（与 `manifest.gen.ts` 的 `vizzes[].id` 一一对应）。 */
export const models: Record<string, SceneModel> = {
  iteration_race: iterationRaceModel,
  error_spectrum: errorSpectrumModel,
  krylov_growth: krylovGrowthModel,
  precond_effect: precondEffectModel,
};

export { manifest } from "./manifest.gen.js";

/** 取某个 viz 的 model；未实现时返回 `null`（不抛错，调度层自行决定如何提示）。 */
export function modelFor(vizId: string): SceneModel | null {
  return Object.prototype.hasOwnProperty.call(models, vizId) ? models[vizId] : null;
}
