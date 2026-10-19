/**
 * 6.2 章「交互式教材打包与视频导出」的前端登记表：viz id → model。
 */

import type { SceneModel } from "../../shared/src/model.js";
import { catalogNavModel } from "./catalog_nav.model.js";
import { exportTimelineModel } from "./export_timeline.model.js";
import { bookBuildModel } from "./book_build.model.js";

export const models: Record<string, SceneModel> = {
  catalog_nav: catalogNavModel,
  export_timeline: exportTimelineModel,
  book_build: bookBuildModel,
};

export { manifest } from "./manifest.gen.js";
export { catalogNavModel } from "./catalog_nav.model.js";
export { exportTimelineModel } from "./export_timeline.model.js";
export { bookBuildModel } from "./book_build.model.js";

export function modelFor(vizId: string): SceneModel | null {
  return Object.prototype.hasOwnProperty.call(models, vizId) ? models[vizId]! : null;
}
