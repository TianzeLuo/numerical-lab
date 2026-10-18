/**
 * 2.5 章的前端装配入口：可视化 id → model 的映射。
 *
 * 这一层刻意**不认识 DOM**（`view.ts` / `*.viz.ts` 才碰 DOM/canvas）：它只回答
 * 「这个 viz 用哪个 model 把帧变成 Scene」。`modelFor` 对未知 id 返回 `null`
 * 而不是抛错——目录里可能有还没实现的 viz，调度层应当自己决定是提示还是跳过。
 *
 * 五个 viz 的分工（对应笔记的五节）：
 *  - `signal_spectrum` 看**定义**：时域与单边谱同步变化，`df`（补零栅格）与 `df_true`（1/T）分开报；
 *  - `leak_window` 看**截断**：非整周期采样如何泄漏，窗的旁瓣/主瓣宽度/相干增益/滚降幂次的账；
 *  - `butterfly_stages` 看**算法**：radix-2 逐级蝶形（唯一带时间轴），位反序靠 |X_k| 峰位钉死；
 *  - `conv_theorem` 看**定理的边界**：频域相乘 = 循环卷积，N < N_a+M−1 时尾部卷绕回来；
 *  - `resolution` 看**极限**：两个近邻音能否分开由 T 决定，补零只加密栅格。
 */

import type { SceneModel } from "../../shared/src/model.js";
import { butterflyStagesModel } from "./butterfly_stages.model.js";
import { convTheoremModel } from "./conv_theorem.model.js";
import { leakWindowModel } from "./leak_window.model.js";
import { resolutionModel } from "./resolution.model.js";
import { signalSpectrumModel } from "./signal_spectrum.model.js";

/** viz id → model（与 `manifest.gen.ts` 的 `vizzes[].id` 一一对应）。 */
export const models: Record<string, SceneModel> = {
  signal_spectrum: signalSpectrumModel,
  leak_window: leakWindowModel,
  butterfly_stages: butterflyStagesModel,
  conv_theorem: convTheoremModel,
  resolution: resolutionModel,
};

export { manifest } from "./manifest.gen.js";

/** 取某个 viz 的 model；未实现时返回 `null`（不抛错，调度层自行决定如何提示）。 */
export function modelFor(vizId: string): SceneModel | null {
  return Object.prototype.hasOwnProperty.call(models, vizId) ? models[vizId] : null;
}
