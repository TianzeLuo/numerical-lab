/**
 * `export_timeline`（6.2 章 viz 2）：视频导出的两本时间账。
 *
 * 1. **内容时刻**：这一帧取自数值解的哪个时刻。最近邻有 h/2 的量化误差；
 *    序号抽稀在非均匀步长下把物理时间拉弯；线性插值把内容时刻做准，但仍有 O(h²) 状态误差。
 * 2. **播放时刻**：GIF 的 delay 以 1/100 s 为单位。24 fps 被写成 4 cs，实际 25 fps，
 *    片尾漂移线性累积；只有 25 fps 这类能整除 100 的帧率才零漂移。
 *
 * 主图画“源步长 + 导出帧落在哪一步”，副图画两条误差曲线；红点标出连续两帧同源的 stall。
 */

import {
  applyOverflow,
  canvasSize,
  dataSourceOf,
  degradedBadge,
  metaNumber,
  palette,
  placeBadge,
  requireArray,
  requireDrawableLayers,
  requirePitfalls,
  requireSameLength,
  sceneShell,
  type SceneModel,
} from "../../shared/src/model.js";
import {
  axisModeBadge,
  badge,
  lineLayer,
  makeAxis,
  scatterLayer,
  series,
  sourceBadge,
} from "../../shared/src/scene.js";
import { axisTitleOf, axisYlabelOf, columnsLayout } from "../_common/panel.js";
import {
  CHAPTER,
  SLOT,
  VIZ_TITLES,
  compactText,
  fmt,
  narrowLayout,
  pushGroups,
  statusLine,
} from "./common.js";
import { manifest } from "./manifest.gen.js";

const VIZ = "export_timeline";
const SPEC = manifest.vizzes.find((v) => v.id === VIZ);
const TITLE = SPEC?.title ?? VIZ_TITLES.export_timeline;

export const exportTimelineModel: SceneModel = (frame, opts) => {
  const pal = palette(opts.theme);
  const { width, height } = canvasSize(opts);
  const sourceStepT = requireArray(frame, "source_step_t");
  const sourceStepH = requireArray(frame, "source_step_h");
  const frameT = requireArray(frame, "frame_t");
  const contentT = requireArray(frame, "content_t");
  const playbackT = requireArray(frame, "playback_t");
  const contentIndex = requireArray(frame, "content_index");
  const timingError = requireArray(frame, "timing_error");
  const playbackError = requireArray(frame, "playback_error");
  const stateError = requireArray(frame, "state_error");
  const stall = requireArray(frame, "stall");
  requireSameLength(sourceStepT, sourceStepH, "source_step_t 与 source_step_h");
  requireSameLength(frameT, contentT, "frame_t 与 content_t");
  requireSameLength(frameT, playbackT, "frame_t 与 playback_t");
  requireSameLength(frameT, contentIndex, "frame_t 与 content_index");
  requireSameLength(frameT, timingError, "frame_t 与 timing_error");
  requireSameLength(frameT, playbackError, "frame_t 与 playback_error");
  requireSameLength(frameT, stateError, "frame_t 与 state_error");
  requireSameLength(frameT, stall, "frame_t 与 stall");
  const pitfalls = requirePitfalls(frame);
  const n = frameT.length;
  if (n < 2 || sourceStepT.length < 1) throw new Error("export_timeline：帧数或源步数不足");

  const tol = metaNumber(frame, "tol");
  const nSource = metaNumber(frame, "n_source");
  const nSteps = metaNumber(frame, "n_steps");
  const fps = metaNumber(frame, "fps");
  const delayCs = metaNumber(frame, "delay_cs");
  const actualFps = metaNumber(frame, "actual_fps");
  const hMin = metaNumber(frame, "h_min");
  const hMax = metaNumber(frame, "h_max");
  const hRatio = metaNumber(frame, "h_ratio");
  const span = metaNumber(frame, "span");
  const nRaw = metaNumber(frame, "n_raw");
  const nRejected = metaNumber(frame, "n_rejected");
  const truncated = (metaNumber(frame, "truncated") ?? 0) > 0.5;
  const nominalDuration = metaNumber(frame, "nominal_duration");
  const playbackDuration = metaNumber(frame, "playback_duration");
  const maxTiming = metaNumber(frame, "max_timing_error");
  const maxPlayback = metaNumber(frame, "max_playback_error");
  const maxState = metaNumber(frame, "max_state_error");
  const nStalls = metaNumber(frame, "n_stalls");
  const maxHold = metaNumber(frame, "max_hold");
  const nSkipped = metaNumber(frame, "n_skipped_source");
  const mode = frame.strings["sampling"] ?? "nearest";

  const scene = sceneShell(frame, opts, {
    chapter: CHAPTER,
    viz: VIZ,
    title: TITLE,
    t: 0,
    dataSource: dataSourceOf(frame),
    pitfalls,
  });
  const degraded: string[] = [];
  const wide = width >= 820;

  let mainRect: [number, number, number, number];
  let auxRect: [number, number, number, number];
  let sidebar: { x: number; w: number } | null = null;
  if (wide) {
    const L = columnsLayout(scene, width, height, { topShare: 0.58, auxH: 150, gap: 56 });
    mainRect = [L.left, L.topY, L.leftW, L.topH];
    auxRect = [L.left, L.botY, L.leftW, Math.min(150, L.botH)];
    sidebar = { x: L.rightX, w: L.rightW };
  } else {
    const N = narrowLayout(scene, width, height, { textRows: 2, mainShare: 0.58, aux: true });
    mainRect = N.main;
    auxRect = N.aux as [number, number, number, number];
    compactText(
      scene,
      N.left,
      N.textTop,
      N.contentW,
      [
        `${fmt(fps, 0)}fps → delay ${fmt(delayCs, 0)}cs / 实际 ${fmt(actualFps, 2)}fps；帧 ${fmt(n, 0)}/${fmt(nRaw, 0)}。`,
        `max|Δt|=${fmt(maxTiming, 3)}s；GIF 漂移=${fmt(maxPlayback, 3)}s；红圈=同源保持 ${fmt(nStalls, 0)} 帧。`,
      ],
      pal.muted,
      mainRect[1] - 2,
    );
    degraded.push("窄画布：省略右栏说明；主图、误差曲线与 stall 红点保留");
  }

  const xMax = span ?? (sourceStepT[sourceStepT.length - 1] as number);
  const hLo = Math.max(1e-12, (hMin ?? 1e-3) * 0.8);
  const hHi = Math.max(hLo * 20, (hMax ?? 1) * 1.25);
  const main = makeAxis("main", mainRect, {
    xlim: [0, Math.max(1e-9, xMax)],
    ylim: [hLo, hHi],
    yscale: "log",
    xlabel: "t",
    ylabel: axisYlabelOf(SPEC?.axes, "main", "步长 h（对数）"),
    title: axisTitleOf(SPEC?.axes, "main", "自适应步长 h(t) 与导出帧"),
    autoscaled: false,
    tickTarget: 5,
  });

  // 帧的位置：x=该帧的内容时刻，y=所用源快照所在步的 h（对数轴只画正步长）。
  const frameX: number[] = [];
  const frameY: number[] = [];
  const stallX: number[] = [];
  const stallY: number[] = [];
  for (let j = 0; j < n; j += 1) {
    const idx = Math.max(0, Math.min(sourceStepH.length - 1, Math.round(contentIndex[j] as number)));
    const h = sourceStepH[idx] as number;
    if (!Number.isFinite(h) || h <= 0) continue;
    frameX.push(frameT[j] as number);
    frameY.push(h);
    if ((stall[j] as number) > 0.5) {
      stallX.push(frameT[j] as number);
      stallY.push(h);
    }
  }

  const lim0 = Math.max(1e-4, ...Array.from(timingError, (v) => Math.abs(v)), ...Array.from(playbackError, (v) => Math.abs(v))) * 1.15;
  const error = makeAxis("error", auxRect, {
    xlim: [0, Math.max(1, n - 1)],
    ylim: [-lim0, lim0],
    xlabel: "帧号 j",
    ylabel: axisYlabelOf(SPEC?.axes, "error", "误差（s）"),
    title: axisTitleOf(SPEC?.axes, "error", "时间误差：内容时刻 / GIF 播放时刻"),
    autoscaled: false,
    tickTarget: 4,
  });
  scene.axes.push(main, error);

  scene.layers.push(
    lineLayer("main", {
      x: sourceStepT,
      y: sourceStepH,
      color: pal.muted,
      width: 1.0,
      alpha: 0.75,
    }),
  );
  // 红圈先画、蓝点后画：stall 的帧保留一圈红色，普通帧是实心蓝点。
  if (stallX.length > 0) {
    scene.layers.push(scatterLayer("main", stallX, stallY, { r: 4.0, color: pal.marker, alpha: 0.95 }));
  }
  scene.layers.push(scatterLayer("main", frameX, frameY, { r: 2.0, color: pal.curves[0] as string, alpha: 0.9 }));

  const idx = Array.from({ length: n }, (_, j) => j);
  scene.layers.push(
    lineLayer("error", [
      series([0, Math.max(1, n - 1)], [0, 0], pal.reference, { width: 0.9, dash: [4, 4] }),
      series(idx, timingError, pal.curves[0] as string, { width: 1.4 }),
      series(idx, playbackError, pal.exact, { width: 1.8, dash: [6, 3] }),
    ]),
  );

  if (sidebar) {
    const verdict =
      mode === "linear"
        ? "线性插值把内容时刻做准；GIF 漂移仍需单独处理。"
        : mode === "decimate"
          ? "序号抽稀把非均匀时间拉平：帧数对，时刻不对。"
          : "最近邻有界抖动；同一快照连续多帧是 stall，不是慢动作。";
    pushGroups(scene, sidebar.x, mainRect[1], auxRect[1] + auxRect[3], sidebar.w, [
      {
        lines: [
          `tol=${fmt(tol, 1)}：${fmt(nSteps, 0)} 步 / ${fmt(nSource, 0)} 点，拒绝 ${fmt(nRejected, 0)} 次。`,
          `h∈[${fmt(hMin, 3)}, ${fmt(hMax, 3)}]；max/min=${fmt(hRatio, 3)}。`,
        ],
        color: pal.ink,
      },
      {
        lines: [
          `GIF：${fmt(delayCs, 0)} cs（${fmt(fps, 0)}→${fmt(actualFps, 2)} fps）。`,
          `名义 ${fmt(nominalDuration, 4)} s；播放 ${fmt(playbackDuration, 4)} s。`,
          `片尾漂移 ${fmt(maxPlayback, 4)} s（25 fps 才为零）。`,
        ],
        color: pal.muted,
      },
      {
        lines: [
          `帧 ${fmt(n, 0)}/${fmt(nRaw, 0)}${truncated ? "（已截断）" : "（未截断）"}。`,
          `stall ${fmt(nStalls, 0)} 帧；最长同源 ${fmt(maxHold, 4)} s。`,
          `未用源快照 ${fmt(nSkipped, 0)} 个。`,
        ],
        color: truncated ? pal.warn : pal.muted,
      },
      {
        lines: [
          `内容 max|Δt|=${fmt(maxTiming, 4)} s。`,
          `帧状态 max|Δy|=${fmt(maxState, 3)}。`,
          `播放 max|Δt|=${fmt(maxPlayback, 4)} s。`,
        ],
        color: pal.muted,
      },
      {
        lines: [
          "蓝=内容误差；橙虚=播放漂移；红圈=stall。",
          verdict,
        ],
        color: pal.note,
      },
    ]);
  }

  scene.badges.push(placeBadge(sourceBadge(dataSourceOf(frame)), SLOT.source, width));
  scene.badges.push(placeBadge(axisModeBadge(false), SLOT.axisMode, width));
  scene.badges.push(
    placeBadge(
      badge(
        "critical",
        `请求 ${fmt(fps, 0)} fps；GIF delay=${fmt(delayCs, 0)} cs；临界 25 fps 才零漂移`,
        8,
        44,
      ),
      SLOT.critical,
      width,
    ),
  );
  scene.badges.push(
    placeBadge(
      badge(
        "status",
        statusLine([
          `${fmt(fps, 0)}→${fmt(actualFps, 2)}fps`,
          `${fmt(n, 0)} 帧`,
          `Δt ${fmt(maxTiming, 3)}s`,
          `漂移 ${fmt(maxPlayback, 3)}s`,
        ]),
        8,
        8,
      ),
      SLOT.status,
      width,
    ),
  );
  const dbg = degradedBadge(degraded, SLOT.degraded, width);
  if (dbg !== undefined) scene.badges.push(dbg);

  applyOverflow(scene, 0, { slot: SLOT.overflow, width });
  requireDrawableLayers(scene);
  return scene;
};
