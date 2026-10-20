import assert from "node:assert/strict";
import { test } from "node:test";

import { sceneForDisplay } from "../shared/src/display.js";
import {
  lineLayer,
  loglogFitLayer,
  makeAxis,
  newScene,
  series,
  sourceBadge,
  validateScene,
  type Scene,
} from "../shared/src/scene.js";

/** 双对数收敛阶图：两条线的文字标注是契约的一部分，显示层不能把它们抹掉。 */
function convergenceScene(): Scene {
  const s = newScene({
    chapter: "3.4",
    viz: "demo_fit",
    title: "收敛阶",
    pitfalls: "只看一条误差曲线容易把斜率读错；必须同时对照理论参考线。",
  });
  s.axes.push(
    makeAxis("main", [60, 40, 500, 300], {
      xscale: "log",
      yscale: "log",
      xlim: [1e-3, 0.1],
      ylim: [1e-6, 0.01],
      xlabel: "h",
      ylabel: "误差",
    }),
  );
  s.layers.push(
    loglogFitLayer("main", [1e-3, 1e-2, 1e-1], [1e-6, 1e-4, 1e-2], 2, 2),
  );
  s.badges.push(sourceBadge(s.dataSource));
  return s;
}

test("显示变换保留 loglogfit 的两条标注，显示场景仍然通过契约校验", () => {
  const raw = convergenceScene();
  assert.deepEqual(validateScene(raw), []);

  const display = sceneForDisplay(raw);
  const fit = display.scene.layers.find((layer) => layer.kind === "loglogfit");
  assert.ok(fit?.kind === "loglogfit", "显示层必须保留拟合图层");
  assert.ok(fit.seriesLabel.length > 0, "实测曲线标注不能为空（规则 2）");
  assert.ok(fit.theoryLabel.length > 0, "理论参考线标注不能为空（规则 2）");

  assert.deepEqual(validateScene(display.scene), [], "显示场景仍必须满足 Scene 契约");
  assert.equal(display.scene.chrome?.texts, false, "显示层关闭所有画布文字");
});

test("显示变换把两条标注同时交给 HTML 图例，且不重复", () => {
  const display = sceneForDisplay(convergenceScene());
  const items = display.legend.filter((item) => item.kind === "loglogfit");
  assert.equal(items.length, 2);
  assert.ok(items.some((item) => /实测斜率/.test(item.label)));
  assert.ok(items.some((item) => /理论斜率/.test(item.label)));
  assert.notEqual(items[0]?.color, items[1]?.color, "实测线与参考线必须用不同颜色区分");
});

test("显示变换不会改坏普通折线的图例收集", () => {
  const raw = convergenceScene();
  raw.axes[0]!.xscale = "linear";
  raw.axes[0]!.xlim = [0, 1];
  raw.axes[0]!.ylim = [0, 1];
  raw.layers = [];
  raw.layers.push({
    kind: "line",
    axes: "main",
    series: [
      { x: [0, 0.5, 1], y: [0, 0.5, 1], color: "#2b6cb0", width: 1.6, label: "迭代解" },
      { x: [0, 0.5, 1], y: [0, 0.6, 1.2], color: "#c53030", width: 1.6, label: "参考解" },
    ],
  });
  const display = sceneForDisplay(raw);
  assert.deepEqual(
    display.legend.map((item) => item.label),
    ["迭代解", "参考解"],
  );
  const line = display.scene.layers[0];
  assert.ok(line?.kind === "line");
  assert.ok(line.series.every((s) => s.label === undefined), "画布上不再重复画内联标签");
});

test("放不下的纵轴名只在 HTML 面板说明里保留，不在画布上伸出面板", () => {
  const raw = newScene({
    chapter: "1.1",
    viz: "axis_label_fit",
    title: "纵轴名排版",
    pitfalls: "窄面板里的纵轴名要完整可读，必要时交给图外说明。",
  });
  raw.axes.push(
    makeAxis("short", [60, 40, 500, 52], {
      xlim: [0, 1],
      ylim: [0, 1],
      ylabel: "该窗口中心间距与邻接浮点数密度",
    }),
    makeAxis("tall", [60, 130, 500, 260], {
      xlim: [0, 1],
      ylim: [0, 1],
      ylabel: "误差",
    }),
  );
  raw.layers.push(
    lineLayer("short", series([0, 1], [0, 1], "#333")),
    lineLayer("tall", series([0, 1], [0, 1], "#333")),
  );
  raw.badges.push(sourceBadge(raw.dataSource));
  assert.deepEqual(validateScene(raw), []);

  const display = sceneForDisplay(raw);
  const shortAxis = display.scene.axes.find((axis) => axis.id === "short");
  const tallAxis = display.scene.axes.find((axis) => axis.id === "tall");
  assert.equal(shortAxis?.ylabel, "", "放不下的纵轴名不应画上画布");
  assert.equal(tallAxis?.ylabel, "误差", "放得下的短纵轴名应当保留");
  assert.equal(
    display.axes.find((axis) => axis.id === "short")?.ylabel,
    "该窗口中心间距与邻接浮点数密度",
    "HTML 面板说明仍然保留完整纵轴名",
  );
});
