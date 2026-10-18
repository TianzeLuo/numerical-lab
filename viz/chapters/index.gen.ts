// 本文件由 tools/gen_manifests.py 从章节模块的 MANIFEST 生成，请勿手改。

import type { ChapterManifestJson } from "../shared/src/manifest.js";
import type { SceneModel } from "../shared/src/model.js";
import { manifest as manifest_c_0_0, models as models_c_0_0 } from "./0.0/index.js";
import { manifest as manifest_c_1_1, models as models_c_1_1 } from "./1.1/index.js";
import { manifest as manifest_c_1_2, models as models_c_1_2 } from "./1.2/index.js";
import { manifest as manifest_c_1_5, models as models_c_1_5 } from "./1.5/index.js";
import { manifest as manifest_c_2_1, models as models_c_2_1 } from "./2.1/index.js";
import { manifest as manifest_c_2_2, models as models_c_2_2 } from "./2.2/index.js";
import { manifest as manifest_c_2_3, models as models_c_2_3 } from "./2.3/index.js";

/** 一章的前端装配：manifest（契约）+ models（viz id → Frame→Scene） */
export interface ChapterModels {
  manifest: ChapterManifestJson;
  models: Record<string, SceneModel>;
}

/** 已经实现可视化的章节（未实现的不在表里，调用方要处理 undefined） */
export const chapterModels: Record<string, ChapterModels> = {
  "0.0": { manifest: manifest_c_0_0, models: models_c_0_0 },
  "1.1": { manifest: manifest_c_1_1, models: models_c_1_1 },
  "1.2": { manifest: manifest_c_1_2, models: models_c_1_2 },
  "1.5": { manifest: manifest_c_1_5, models: models_c_1_5 },
  "2.1": { manifest: manifest_c_2_1, models: models_c_2_1 },
  "2.2": { manifest: manifest_c_2_2, models: models_c_2_2 },
  "2.3": { manifest: manifest_c_2_3, models: models_c_2_3 },
};

export default chapterModels;
