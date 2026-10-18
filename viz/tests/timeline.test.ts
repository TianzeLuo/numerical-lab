/**
 * 时间轴控制器的测试：注入时钟，所以"播放 10 秒"只是几次函数调用。
 *
 * 重点在两件容易做错的事：
 * - **限流与零头**：`fps` 是"最多多快"，不足一帧的时间必须攒着，否则长期速度会偏慢；
 * - **丢帧**：真实时间跨过 5 个帧间隔时只把 `t` 推到最新，跳过几帧就记几笔 `droppedFrames`
 *   （UI 可以显示"因来不及而丢掉 N 帧"，而不是假装一切正常）。
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { Timeline } from "../shared/src/timeline.js";
import type { TimelineSnapshot } from "../shared/src/timeline.js";

/** 假时钟：测试自己决定"过了多久"。 */
class Clock {
  ms = 0;
  readonly now = (): number => this.ms;
  advance(ms: number): void {
    this.ms += ms;
  }
}

function makeTimeline(overrides: Partial<{ mode: "none" | "stream" | "scrub"; tMax: number; dt: number; fps: number }> = {}): {
  tl: Timeline;
  clock: Clock;
  events: TimelineSnapshot[];
} {
  const clock = new Clock();
  const tl = new Timeline({
    mode: overrides.mode ?? "stream",
    tMax: overrides.tMax ?? 1,
    dt: overrides.dt ?? 0.1,
    fps: overrides.fps ?? 10,
    now: clock.now,
  });
  const events: TimelineSnapshot[] = [];
  tl.onChange((s) => events.push(s));
  return { tl, clock, events };
}

test("初始状态：t=0、未播放、帧数 = floor(tMax/dt)+1", () => {
  const { tl } = makeTimeline({ tMax: 1, dt: 0.1 });
  assert.equal(tl.t, 0);
  assert.equal(tl.playing, false);
  assert.equal(tl.rate, 1);
  assert.equal(tl.frameCount, 11);
  assert.equal(tl.frameIndex, 0);
  assert.equal(tl.progress, 0);
  assert.equal(tl.droppedFrames, 0);
});

test("play/pause/toggle 改变播放态并触发 onChange", () => {
  const { tl, events } = makeTimeline();
  tl.play();
  assert.equal(tl.playing, true);
  assert.equal(events.length, 1);
  tl.play(); // 幂等：不重复通知
  assert.equal(events.length, 1);
  tl.pause();
  assert.equal(tl.playing, false);
  tl.toggle();
  assert.equal(tl.playing, true);
  tl.toggle();
  assert.equal(tl.playing, false);
  assert.equal(events.length, 4);
  assert.equal(events[0].playing, true);
  assert.deepEqual(Object.keys(events[0]).sort(), ["dropped", "ended", "frame", "frames", "playing", "rate", "t"]);
});

test("advance：受 fps 限流，不足一帧不推进（且零头会攒着）", () => {
  const { tl, clock } = makeTimeline({ fps: 10, dt: 0.1 }); // 一帧 100 ms
  tl.play();

  clock.advance(60);
  assert.equal(tl.advance(), 0);
  assert.equal(tl.t, 0);

  clock.advance(60); // 累计 120 ms → 应推进 1 帧，剩 20 ms
  assert.equal(tl.advance(), 1);
  assert.equal(tl.t, 0.1);

  clock.advance(60); // 20 + 60 = 80 ms，还不够
  assert.equal(tl.advance(), 0);
  clock.advance(40); // 120 ms → 再推进 1 帧
  assert.equal(tl.advance(), 1);
  assert.equal(tl.t, 0.2);
});

test("advance：跨过多个帧间隔时只推到最新，跳过几帧就记几笔丢帧", () => {
  const { tl, clock } = makeTimeline({ fps: 10, dt: 0.1, tMax: 100 });
  tl.play();
  clock.advance(550); // 5.5 帧
  assert.equal(tl.advance(), 5);
  assert.equal(tl.t.toFixed(1), "0.5");
  assert.equal(tl.droppedFrames, 4, "5 帧里只有 1 帧被画出来");
  assert.equal(tl.progress, 0.005);
});

test("advance：没在播放 / 时钟没走 → 不推进", () => {
  const { tl, clock } = makeTimeline({ fps: 10 });
  clock.advance(1000);
  assert.equal(tl.advance(), 0, "没播放");
  tl.play();
  assert.equal(tl.advance(), 0, "刚 play，时钟没走");
  clock.advance(100);
  assert.equal(tl.advance(), 1);
});

test("setRate：2× 速度按比例推进；非法速度回落到 1×", () => {
  const { tl, clock } = makeTimeline({ fps: 10, dt: 0.1, tMax: 100 });
  tl.setRate(2);
  assert.equal(tl.rate, 2);
  tl.play();
  clock.advance(100); // 100 ms × 2 = 200 ms → 2 帧
  assert.equal(tl.advance(), 2);
  assert.equal(tl.t.toFixed(1), "0.2");

  tl.setRate(0);
  assert.equal(tl.rate, 1, "0 或负数回落到 1×");
  tl.setRate(Number.NaN);
  assert.equal(tl.rate, 1);
  tl.setRate(999);
  assert.equal(tl.rate, 16, "上限 16×");
});

test("seek：夹到 [0, tMax]，到末端标记 ended，并重置节流", () => {
  const { tl, clock } = makeTimeline({ fps: 10, dt: 0.1, tMax: 1 });
  tl.play();
  clock.advance(100);
  tl.seek(0.35);
  assert.equal(tl.t, 0.35);
  assert.equal(tl.frameIndex, 4, "round(0.35/0.1)=4（浮点噪声被吸收）");
  clock.advance(100); // seek 重置了节流基准
  assert.equal(tl.advance(), 1);
  assert.equal(tl.t.toFixed(2), "0.45");

  tl.seek(99);
  assert.equal(tl.t, 1);
  assert.equal(tl.ended, true);
  tl.seek(-5);
  assert.equal(tl.t, 0);
  assert.equal(tl.ended, false);
});

test("step：单步（可负），不改变播放态也不产生丢帧", () => {
  const { tl } = makeTimeline({ mode: "scrub", tMax: 1, dt: 0.1 });
  tl.step(3);
  assert.equal(tl.t.toFixed(1), "0.3");
  tl.step(-1);
  assert.equal(tl.t.toFixed(1), "0.2");
  tl.step(1000);
  assert.equal(tl.t, 1, "夹到 tMax");
  assert.equal(tl.droppedFrames, 0, "单步不是丢帧");
  tl.step(0);
  assert.equal(tl.t, 1);
});

test("到 tMax 自动暂停，并把最后一段没画的帧记进丢帧", () => {
  const { tl, clock } = makeTimeline({ fps: 10, dt: 0.1, tMax: 0.5 });
  tl.play();
  clock.advance(10_000); // 一口气跑完
  const advanced = tl.advance();
  assert.equal(tl.t, 0.5);
  assert.equal(tl.playing, false, "到末端自动停");
  assert.equal(tl.ended, true);
  assert.equal(advanced, 5, "从 0 到 0.5 共 5 帧");
  assert.equal(tl.droppedFrames, 4);

  tl.play(); // 在末端按播放：从头重播
  assert.equal(tl.t, 0);
  assert.equal(tl.ended, false);
  assert.equal(tl.playing, true);
});

test("scrub 模式：play() 无效，只有 seek/step 能改 t", () => {
  const { tl, clock } = makeTimeline({ mode: "scrub", tMax: 1, dt: 0.1 });
  tl.play();
  assert.equal(tl.playing, false, "scrub 不播放");
  clock.advance(1000);
  assert.equal(tl.advance(), 0);
  assert.equal(tl.t, 0);
  tl.seek(0.4);
  tl.step(1);
  assert.equal(tl.t.toFixed(1), "0.5");
});

test("none 模式：没有时间轴，一切推进都是 no-op", () => {
  const { tl, clock } = makeTimeline({ mode: "none", tMax: 5, dt: 0.1 });
  assert.equal(tl.tMax, 0, "none 模式忽略 tMax");
  assert.equal(tl.frameCount, 1);
  tl.play();
  tl.seek(3);
  tl.step(5);
  clock.advance(1000);
  assert.equal(tl.advance(), 0);
  assert.equal(tl.t, 0);
  assert.equal(tl.playing, false);
  assert.equal(tl.progress, 0);
});

test("onChange 返回退订函数；订阅者抛错不影响时间轴", () => {
  const { tl } = makeTimeline();
  let hits = 0;
  const off = tl.onChange(() => {
    hits++;
  });
  tl.onChange(() => {
    throw new Error("订阅者自己的 bug");
  });
  tl.play();
  assert.equal(hits, 1);
  off();
  tl.pause();
  assert.equal(hits, 1, "退订后不再收到通知");
});

test("reset：回到起点并清空丢帧计数", () => {
  const { tl, clock } = makeTimeline({ fps: 10, dt: 0.1, tMax: 1 });
  tl.play();
  clock.advance(900);
  tl.advance();
  assert.ok(tl.droppedFrames > 0);
  tl.reset();
  assert.equal(tl.t, 0);
  assert.equal(tl.playing, false);
  assert.equal(tl.droppedFrames, 0);
  assert.equal(tl.ended, false);
});

test("容错：非法 dt/fps 不会让时间轴失控（回落到安全值）", () => {
  const clock = new Clock();
  const tl = new Timeline({ mode: "stream", tMax: 1, dt: 0, fps: Number.NaN, now: clock.now });
  assert.equal(tl.dt, 1);
  assert.equal(tl.fps, 30);
  tl.play();
  clock.advance(1000 / 30);
  assert.equal(tl.advance(), 1);
  assert.equal(tl.t, 1);
});
