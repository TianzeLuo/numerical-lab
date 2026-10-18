/**
 * 2.2 章的前端装配入口：可视化 id → model 的映射。
 *
 * 这一层刻意**不认识 DOM**（`view.ts` / `*.viz.ts` 才碰 DOM/canvas）：它只回答
 * 「这个 viz 用哪个 model 把帧变成 Scene」。`modelFor` 对未知 id 返回 `null`
 * 而不是抛错——目录里可能有还没实现的 viz，调度层应当自己决定是提示还是跳过。
 *
 * 四个 viz 的分工：`control_points` 看**参数**（bc 与 τ 怎么改曲线、改的是哪条连续性）、
 * `smoothness_c0c1c2` 看**接缝**（C⁰/C¹/C² 的二阶导在各节点上长什么样）、
 * `bspline_local` 看**局部性**（改一个控制点动多宽、插节点为什么不改曲线）、
 * `spline_vs_poly` 看**稳定性**（同一组数据上多项式爆炸、样条收敛，以及舍入地板）。
 */

import type { SceneModel } from "../../shared/src/model.js";
import { bsplineLocalModel } from "./bspline_local.model.js";
import { controlPointsModel } from "./control_points.model.js";
import { splineVsPolyModel } from "./spline_vs_poly.model.js";
import { smoothnessModel } from "./smoothness_c0c1c2.model.js";

export const models: Record<string, SceneModel> = {
  control_points: controlPointsModel,
  smoothness_c0c1c2: smoothnessModel,
  bspline_local: bsplineLocalModel,
  spline_vs_poly: splineVsPolyModel,
};

export { manifest } from "./manifest.gen.js";

/** 取某个 viz 的 model；未实现时返回 `null`（不抛错，调度层自行决定如何提示）。 */
export function modelFor(vizId: string): SceneModel | null {
  return Object.prototype.hasOwnProperty.call(models, vizId) ? models[vizId] : null;
}
