/**
 * 3.5 章的前端装配入口：可视化 id → model 的映射。
 *
 * 这一层刻意**不认识 DOM**（`view.ts` / `*.viz.ts` 才碰 DOM/canvas）：它只回答
 * 「这个 viz 用哪个 model 把帧变成 Scene」。`modelFor` 对未知 id 返回 `null`
 * 而不是抛错——目录里可能有还没实现的 viz，调度层应当自己决定是提示还是跳过。
 *
 * 五个 viz 的分工：`lorenz_butterfly` 看**可预报时间**（1e-8 的差要多久才看得见）、
 * `bifurcation_logistic` 看**周期窗口与分辨率**（拖 r 实时生成）、
 * `lyapunov_overlay` 看**指数本身有多可信**（分块标准误 + 三个解析对拍）、
 * `poincare_section` 看**降维的代价**（线性插值点 vs Newton 精化点）、
 * `phase_portrait` 看**相型**（零倾线、平衡点分类、极限环半径 vs √μ）。
 */

import type { SceneModel } from "../../shared/src/model.js";
import { bifurcationLogisticModel } from "./bifurcation_logistic.model.js";
import { lorenzButterflyModel } from "./lorenz_butterfly.model.js";
import { lyapunovOverlayModel } from "./lyapunov_overlay.model.js";
import { phasePortraitModel } from "./phase_portrait.model.js";
import { poincareSectionModel } from "./poincare_section.model.js";

/** viz id → model（与 `manifest.gen.ts` 的 `vizzes[].id` 一一对应）。 */
export const models: Record<string, SceneModel> = {
  lorenz_butterfly: lorenzButterflyModel,
  bifurcation_logistic: bifurcationLogisticModel,
  lyapunov_overlay: lyapunovOverlayModel,
  poincare_section: poincareSectionModel,
  phase_portrait: phasePortraitModel,
};

export { manifest } from "./manifest.gen.js";

/** 取某个 viz 的 model；未实现时返回 `null`（不抛错，调度层自行决定如何提示）。 */
export function modelFor(vizId: string): SceneModel | null {
  return Object.prototype.hasOwnProperty.call(models, vizId) ? models[vizId] : null;
}
