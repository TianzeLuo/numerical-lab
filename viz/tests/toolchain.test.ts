import { test } from "node:test";
import assert from "node:assert/strict";

/**
 * 工具链金丝雀：不依赖任何项目模块，只验证
 * tsc 编译 → dist/viz 下的原生 ESM → node:test 这条链路是通的。
 * 如果这个测试挂了，说明是构建配置的问题，而不是某个章节的问题。
 */
test("toolchain: ESM + node:test 可用", () => {
  const xs = [1, 2, 3];
  assert.equal(xs.reduce((a, b) => a + b, 0), 6);
  assert.ok(typeof globalThis.structuredClone === "function", "需要 Node ≥ 17");
});
