/**
 * `viz/shared/src/numeric/` 汇总导出。
 *
 * **本目录只服务 G7 双实现对拍，不是渲染路径。**
 * 目录内容与 PROTOCOLS §7 的"覆盖清单"一一对应，且必须遵守两条纪律：
 *
 * 1. 只放纯函数：无 IO、无 DOM、无第三方依赖（P7 的精神；`viz/tests/**` 才允许 `node:*`）；
 * 2. `viz/chapters/**` MUST NOT 直接 import 本目录（P8）——章节前端只能消费后端帧，
 *    或在标注 `preview-approx` 后调用**已通过对拍**的函数。
 *
 * 新增算法时，必须同时在 `tools/parity.py::CASE_SPECS` 里登记用例、
 * 在 `tests/parity/tolerances.yaml` 里给出该 case 的容差与理由，否则 G7 不会覆盖它。
 */

export * from "./heat.js";
export * from "./oscillator.js";
export * from "./orderfit.js";
