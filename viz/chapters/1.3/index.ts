/**
 * 1.3 章的前端装配入口：可视化 id → model 的映射。
 *
 * 这一层刻意**不认识 DOM**（`view.ts` / `*.viz.ts` 才碰 DOM/canvas）：它只回答
 * "这个 viz 用哪个 model 把帧变成 Scene"。`modelFor` 对未知 id 返回 `null`
 * 而不是抛错——目录里可能有还没实现的 viz，调度层应当自己决定是提示还是跳过。
 *
 * 四个 viz 的分工：`projection_3d` 看**几何**（b 投影到列空间、残差为什么垂直）、
 * `normal_vs_qr` 看**误差**（正规方程的 κ² vs QR 的 κ，以及"残差小 ≠ 解准"）、
 * `svd_truncation` 看**谱与截断**（L 曲线、Picard 条件、最优 k）、
 * `uncertainty_fan` 看**不确定度**（等残差的谷底有多长、残差对解的移动有多瞎）。
 */

import type { SceneModel } from "../../shared/src/model.js";
import { normalVsQrModel } from "./normal_vs_qr.model.js";
import { projection3dModel } from "./projection_3d.model.js";
import { svdTruncationModel } from "./svd_truncation.model.js";
import { uncertaintyFanModel } from "./uncertainty_fan.model.js";

/** viz id → model（与 `manifest.gen.ts` 的 `vizzes[].id` 一一对应）。 */
export const models: Record<string, SceneModel> = {
  projection_3d: projection3dModel,
  normal_vs_qr: normalVsQrModel,
  svd_truncation: svdTruncationModel,
  uncertainty_fan: uncertaintyFanModel,
};

export { manifest } from "./manifest.gen.js";

/** 取某个 viz 的 model；未实现时返回 `null`（不抛错，调度层自行决定如何提示）。 */
export function modelFor(vizId: string): SceneModel | null {
  return Object.prototype.hasOwnProperty.call(models, vizId) ? models[vizId] : null;
}
