## 章节 6.2 数值断言报告

> 本表由 `tools/run_asserts.py` 从实际运行结果生成，**不是手写的**。

- 断言条数：18（A 0 / B 18 / C 0）
- 通过：18，失败：0
- 声明文件：`chapters/6.2/asserts.yaml`

| 类别 | 断言 | 理论值 | 实测值 | 容差 | 判定 |
|---|---|---|---|---|---|
| B | 三页默认帧的有限性、装帧与 requiredAux | 三页默认帧的数组/meta 全部有限；pack_frame 后 JSON 头无 NaN/Infinity；requiredAux 可定位 | 非有限 JSON 常量 0 个；catalog_nav: 17 数组/14 meta；export_timeline: 16 数组/28 meta；book_build: 13 数组/18 meta | 非有限常量=0，requiredAux 全部命中 | ✅ |
| B | 导出帧与核心时间轴逐位对拍 | 帧数组与直接调用 integrate_adaptive+export_plan 的结果逐位相等；meta 的帧数/delay/漂移与计划一致 | 8 个数组最大差=0.0e+00；n=97，delay=4cs，max\|Δt\|=0.41841，GIF 漂移=0.16000 s | 全部逐位相等；meta 精确 | ✅ |
| B | 最近邻/线性/序号抽稀与 GIF 帧率的独立证据 | linear 的内容时刻误差=0；decimate 的时间误差 > nearest；linear 状态误差 < decimate；fps=25 的 GIF 漂移为 0 | max\|Δt\|：nearest=0.41841，linear=0.0e+00，decimate=1.11667；max\|Δy\|：linear=8.39e-03，decimate=2.30e-01；GIF 漂移：24fps=0.1600s，25fps=0.0e+00s | linear timing=0；decimate>2×nearest；linear state<decimate/4；25fps 零漂移而 24fps>0.1s | ✅ |
| B | 目录帧与展开后的依赖图对拍 | 节点=28、边=展开后的依赖总数；四项错误全 0；焦点 4.3 的祖先闭包=8 章、9 条边；最长链 9 章 | 节点 28、边 67（期望 67）；闭包 8 章、链长 9；错误计数 [0.0, 0.0, 0.0, 0.0] | 节点=28、边=67、闭包=8、链长=9、错误=0 | ✅ |
| B | 分部/焦点高亮掩码 | part=IV 时高亮 6 个节点、其余为 0；focus=all 时全部高亮；焦点掩码恰有一个 1 | part=IV 高亮 6 节点、22 条边；all=28；焦点标记 1 | IV=6 节点、all=28、焦点标记=1 | ✅ |
| B | 讲义审计矩阵与磁盘实际内容对拍 | 每个 0/1 格子都由 notes/附表/报告条数/引用/输入/字形/pitfalls 独立复算；问题数=6−行和 | 28 章 × 6 项；不一致 0 处；总问题数 0（meta 0） | 所有格子与独立复算一致；问题总数吻合 | ✅ |
| B | 讲义的部筛选与失败优先排序 | part=I 只留 5 章；fail 排序把问题多的章放前面（同分按目录序）；行序与 chapter_ids 对应 | part=I 顺序=['1.1', '1.2', '1.3', '1.4', '1.5']；fail 顺序=['1.1', '1.2', '1.3', '1.4', '1.5']，问题数=[np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0)] | 筛选 5 章；fail 的问题数非增；两页章集合一致 | ✅ |
| B | 6.2 manifest 的参数/临界值/requiredAux 契约 | 3 个 viz 的 id 固定；每页 requiredAux 在默认帧里存在；fps 参数声明临界值 25 且 criticalNote 非空 | catalog_nav: params=2, aux=('error',)；export_timeline: params=4, aux=('error',)；book_build: params=2, aux=('error',)；fps.critical=(25.0,) | 3 页齐备、requiredAux 命中、fps 临界值 25 有说明 | ✅ |
| B | 导出帧数、边界与时长账本 | n=floor(fps·(t_end−t_0))+1；frame_t 严格递增且不越界；名义/播放总时长与 delay_cs 精确定义 | 12/nearest: n=49, raw=49；12/linear: n=49, raw=49；12/decimate: n=49, raw=49；24/nearest: n=97, raw=97；24/linear: n=97, raw=97（共 15 组） | 全部 15 组：帧数公式精确、frame_t 严格递增、时长公式精确 | ✅ |
| B | GIF delay 厘秒量化账本 | delay_cs=max(1,round(100/fps))；actual_fps=100/delay_cs；25 fps 精确，24/30/60 都会被改写 | 12→8cs/12.5fps、24→4cs/25fps、25→4cs/25fps、30→3cs/33.33fps、60→2cs/50fps；25fps 每帧误差=0，24fps 每帧 -1/600 s | delay/actual_fps 与公式逐项相等；25fps 零漂移 | ✅ |
| B | 最近邻取样的内容时刻误差上界 | 最近邻内容时刻误差 \|content_t−frame_t\| ≤ max(h)/2（落在最近快照的 Voronoi 半格内） | max\|Δt\|=0.500000；上界 hmax/2=0.500000（hmax=1.0000）；12:0.50000、24:0.50000、25:0.48000、30:0.50000、60:0.50000 | worst ≤ hmax/2 + 1e-12 | ✅ |
| B | 线性重采样的时间精确性与二阶状态误差 | linear 的内容时刻误差=0；状态误差 ≤ \|y''\|max·hmax²/8，logistic 的 \|y''\|max=0.0962 | max\|timing\|=0.000e+00；max\|Δy\|=7.4738e-03（界 1.2100e-02）；最近邻 max\|Δy\|=6.1151e-02 | timing=0（≤1e-15）；state ≤ 0.0121·hmax²；且优于最近邻 4 倍 | ✅ |
| B | 按序号抽稀在非均匀时间轴上的失真 | decimate 的帧数不变、源下标单调非降且末帧取到 n−1；其时间误差显著大于最近邻（非均匀步长被拉平） | nearest max\|Δt\|=0.50000，decimate=1.55000（3.10×）；下标单调=True、末帧取 n−1=True、帧数相同=True | decimate > 2×nearest；下标单调且末帧覆盖 | ✅ |
| B | 帧数上限截断的显式账本 | max_frames=K 时实出 K 帧、truncated=True、n_raw−K 帧被显式记账；末帧时刻 < t_end（跨度为整帧时除外） | raw=97，budget=10，实出=10，截断=87 帧，末帧 t=0.3750 < 4.0000 | 实出=budget、truncated=True、末帧明显提前 | ✅ |
| B | 目录依赖的拓扑序合法性 | 拓扑序包含全部 28 章；每条依赖边 u→v 中 u 在 v 前；6.2（deps=all）必须最后 | 28 章、67 条边；环 0；违例 0；末位=6.2 | 28 章全部出现、0 环、0 违例、末位 6.2 | ✅ |
| B | 依赖分层与关键路径 | depth(v)=1+max depth(dep)；最长链长度=max depth+1=9，且链上相邻两项确实有依赖边 | 9 层、max depth=8；链长 9：1.1 → 1.2 → 1.4 → 1.5 → 4.1 → 4.2 → 4.3 → 6.1 → 6.2 | 分层是划分；depth 递推精确；链长=max depth+1 且末位 6.2 | ✅ |
| B | all 展开与非法依赖的显式报告 | all 展开为清单中位于其前的所有章（不含自身）；未知依赖/自依赖只进问题列表、不静默变成边 | deps={'1.1': (), '1.2': ('1.1',), '1.3': ('1.1', '1.2')}；问题=['1.3: 未知依赖 9.9', '1.3: 自依赖'] | 1.3 依赖 1.1/1.2；未知与自依赖各报 1 条，不进入 deps | ✅ |
| B | 注释剔除、重复标签与附表行计数 | 注释中的 \ref/\input/\label 不计；重复 label 被单独标出；附表行只数 A/B/C & 开头的正文行 | labels=('sec:a', 'sec:a')，refs=('sec:a', 'eq:b')，inputs=('chapters/x/asserts_table',)，dups=('sec:a',)，rows=3 | 注释中的 3 个命令全部忽略；重复 label 检出；附表正文 3 行 | ✅ |
