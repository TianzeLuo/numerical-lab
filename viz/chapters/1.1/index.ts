/**
 * 1.1 章的前端装配入口：可视化 id → model 的映射。
 *
 * 这一层刻意**不认识 DOM**（`view.ts` / `*.viz.ts` 才碰 DOM/canvas）：它只回答
 * 「这个 viz 用哪个 model 把帧变成 Scene」。`modelFor` 对未知 id 返回 `null`
 * 而不是抛错——目录里可能有还没实现的 viz，调度层应当自己决定是提示还是跳过。
 *
 * 四个 viz 的分工（与 `server/chapters/1.1.py` 的 MANIFEST 一一对应）：
 * `float_zoom` 看**表示**（网格有多细、0.1+0.2 落在哪一格）、
 * `cancellation` 看**传播**（输入误差被放大多少、减法本身有多准）、
 * `cond_ellipsoid` 看**问题**（κ 决定的几何与界）、
 * `summation_order` 看**算法**（顺序能救回多少、Kahan 的界与它的边界）。
 */

import type { SceneModel } from "../../shared/src/model.js";
import { cancellationModel } from "./cancellation.model.js";
import { condEllipsoidModel } from "./cond_ellipsoid.model.js";
import { floatZoomModel } from "./float_zoom.model.js";
import { summationOrderModel } from "./summation_order.model.js";

/** viz id → model（与 `manifest.gen.ts` 的 `vizzes[].id` 一一对应）。 */
export const models: Record<string, SceneModel> = {
  float_zoom: floatZoomModel,
  cancellation: cancellationModel,
  cond_ellipsoid: condEllipsoidModel,
  summation_order: summationOrderModel,
};

export { manifest } from "./manifest.gen.js";

/** 取某个 viz 的 model；未实现时返回 `null`（不抛错，调度层自行决定如何提示）。 */
export function modelFor(vizId: string): SceneModel | null {
  return Object.prototype.hasOwnProperty.call(models, vizId) ? models[vizId] : null;
}
