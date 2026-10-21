r"""6.2 章：交互式教材打包与视频导出（3 个可视化）。

这一章不再解方程，而是审计**产出链路**本身。三个页面各回答一个"看起来成功"背后的
可测量问题：

1. ``catalog_nav`` —— 28 章的目录不是按章节号排出来的，而是依赖图的最长路径分层。
   主图是依赖 DAG（节点=章，边=前置依赖），副图是四项图错误的计数；选中的章会高亮
   它的祖先闭包。``deps: [all]`` 被展开为"清单里排在它前面的所有章"。
2. ``export_timeline`` —— 视频的时间轴有**两本账**：内容时刻（这一帧取自数值解的哪个
   时刻）与播放时刻（GIF 的 delay 以 1/100 s 为单位）。最近邻取样的内容误差被半个
   源步长界住，而 24 fps 的 GIF 实际按 25 fps 播放，漂移随帧号线性累积。
3. ``book_build`` —— 讲义构建的逐章检查：笔记、附表行数与声明一致、引用闭合、
   ``\input`` 目标存在、字形无风险、pitfalls 达标。热图的行=章、列=检查项；
   副图是每章的问题数。页面不把"编译没报错"当成"内容闭合"。

三个页面的数据都来自确定性计算（:mod:`numcore.apps.packaging` + 当前仓库的文本/图），
不读 ``dist/`` 的临时报告——否则"审计"会随上一次构建的残留而漂移。
"""

from __future__ import annotations

import functools
import json
import math
import pathlib
import re
from typing import Any, Mapping, Sequence

import numpy as np

from numcore.apps import packaging
from numcore.ode import rk
from numcore.ode.stepsize import integrate_adaptive
from server.nlf import Frame

__all__ = ["MANIFEST", "compute"]

CHAPTER = "6.2"
_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PARTS = ("I", "II", "III", "IV", "V", "VI")
_CHECK_IDS = ("notes", "table", "refs", "inputs", "glyph", "pitfalls")
_CHECK_LABELS = ("笔记", "附表", "引用", "输入", "字形", "陷阱")
_ROMAN = {p: i + 1 for i, p in enumerate(_PARTS)}


def _viz(**kwargs: Any) -> dict[str, Any]:
    return kwargs


_PITFALLS = {
    "catalog_nav": (
        "目录顺序不是构建顺序：章节号只反映写作计划，真正能编译的前提是依赖已就位。"
        "把 3.4 放在 2.3 前面会缺正交多项式，把 6.1 放在 4.x 前面会缺四个求解器。"
        "第二：``deps: [all]`` 不是「依赖所有章」的字面意思，本章把它展开为「清单里排在"
        "它前面的所有章」，否则 6.2 会依赖自己。第三：没有环 ≠ 顺序唯一；同层节点按"
        "清单顺序稳定排序，换实现若用集合遍历，构建顺序会漂移。"
        "第四：图的边数不等于依赖条目数——``all`` 会展开成 27 条边。"
    ),
    "export_timeline": (
        "「视频能播」不等于「时间轴对」。内容时刻与播放时刻是两本账：最近邻取样把每帧"
        "量化到源快照，误差有半个源步长的上界；按序号抽稀则在非均匀步长下把物理时间"
        "拉弯。第二：GIF 的 delay 是整数厘秒，24 fps 请求被写成 4 cs，实际播放 25 fps，"
        "60 fps 被写成 2 cs，实际只有 50 fps；误差随帧号线性累积，长时间片尾会明显快/慢。"
        "第三：帧数上限截断不是「短一点」——它直接改变导出的时间跨度，必须在账本里报出来。"
        "第四：最近邻内容误差小，不代表画面不抖；同一快照被连续显示多帧（stall）是"
        "另一条独立的量。"
    ),
    "book_build": (
        "「编译通过」不是「书对」：xelatex 对缺字形只吞字、不报错，未定义引用也要两遍"
        "编译才稳定；附表的行数如果与断言报告的条数不一致，PDF 里就是旧数字。"
        "第二：引用必须在单章内闭合，因为 G9 逐章编译；跨章 eqref 引用在合并本里能过，"
        "单章编译会失败。第三：注释里的 input/ref 命令不算依赖，先剔注释再计数。"
        "第四：6 列检查里任何一格为 0 都不能靠「其它列全绿」抵消；问题数按章统计，"
        "不是全局平均。"
    ),
}


MANIFEST: dict[str, Any] = {
    "chapter": CHAPTER,
    "title": "交互式教材打包与视频导出",
    "part": "VI",
    "summary": (
        "把 28 章组织成可导航、可导出、可编译的产品：目录依赖图决定构建顺序，"
        "视频导出的内容时刻与 GIF 播放时刻必须分开记账，讲义构建则逐章核对引用闭合、"
        "附表行数、输入文件与字形风险。三个页面都在审计「看起来成功」背后的可测量不变量。"
    ),
    "vizzes": [
        _viz(
            id="catalog_nav",
            title="目录与依赖：28 章的构建顺序",
            description=(
                "节点是章、边是前置依赖，按最长路径分层。选中某一章会高亮它的祖先闭包；"
                "选中某一部会高亮该部的节点。副图是环、未知依赖、自依赖、缺模块四项"
                "错误计数（都应为 0）。侧栏报最长关键路径与闭包大小。"
            ),
            dataSource="backend",
            params=[
                {
                    "name": "part",
                    "label": "高亮分部",
                    "type": "enum",
                    "options": ["all", "I", "II", "III", "IV", "V", "VI"],
                    "default": "all",
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "选择要高亮的分部（I 到 VI，all 表示不按分部筛选）。选中某一部后，"
                        "该部的节点会着色，只要边有一端属于该部也会高亮，其他章仍以灰色保留，"
                        "方便看出跨部依赖；选 all 则回到按焦点章的前置闭包高亮。它和焦点章会互相覆盖："
                        "分部不是 all 时，闭包高亮让位给分部颜色，但焦点章仍有红圈、侧栏的闭包章数仍按焦点计算。"
                        "先看主图的颜色覆盖了哪一层，再看右栏的焦点说明。"
                    ),
                },
                {
                    "name": "focus",
                    "label": "焦点章（高亮其前置闭包）",
                    "type": "enum",
                    "options": ["all", *[
                        "1.1", "1.2", "1.3", "1.4", "1.5",
                        "2.1", "2.2", "2.3", "2.4", "2.5", "2.6",
                        "3.1", "3.2", "3.3", "3.4", "3.5",
                        "4.1", "4.2", "4.3", "4.4", "4.5", "4.6",
                        "5.1", "5.2", "5.3", "5.4", "6.1", "6.2",
                    ]],
                    "default": "4.3",
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "选择要追查前置依赖的章，all 表示显示全图。选中某一章后，图上会高亮它自身以及"
                        "所有直接、间接的前置章，红圈只标焦点章本身，侧栏报闭包大小；默认 4.3 的闭包是 8 章。"
                        "切到 all 会取消闭包高亮；把分部切到某一部时闭包高亮暂停，但红圈与闭包章数仍按焦点计算。"
                        "读图时先找红圈，再沿着高亮边往回走，看看这一章到底依赖了哪些章。"
                    ),
                },
            ],
            time={"mode": "none"},
            requiredAux=["error"],
            axes=[
                {
                    "id": "main",
                    "autoscaled": False,
                    "title": "依赖图（按最长路径分层；横轴 $\\operatorname{depth}$）",
                    "ylabel": "层内序号（仅用于排版）",
                },
                {
                    "id": "error",
                    "autoscaled": False,
                    "title": "依赖图的错误计数（四格都应为 $0$）",
                    "ylabel": "计数（个）",
                },
            ],
            pitfalls=_PITFALLS["catalog_nav"],
            theory=[
                "**先把清单变成有向图。** 打开 docs/chapters.yaml，每一章是一个节点，"
                "deps 里的每个章名是一条“必须先构建”的有向边：$u\\to v$ 表示 $u$ 必须排在 $v$ 前面。"
                "如果写 all，就展开成清单里排在它前面的所有章，不包含自身；未知章名、自依赖不会悄悄丢掉，"
                "而是记进问题清单，重复依赖只保留第一次出现的顺序。当前清单得到 $|V|=28$ 个节点、"
                "$|E|=67$ 条边。常见的误读是按章节号排序就能编译：章节号只反映写作计划，真正的前提是依赖"
                "已经就位；另外 6.2 的 all 展开成 27 条入边，而不是 28 条。",
                "**无环是拓扑序存在的条件。** 拓扑序是一个线性顺序，使每条边都从前指向后。若存在环 "
                "$v_1\\to v_2\\to\\ldots\\to v_k\\to v_1$，环上每个节点都有一条来自环内的入边，任何线性"
                "顺序都无法让所有边朝前，所以拓扑序不存在。反过来，有限无环图一定有入度为 $0$ 的起点："
                "从任意节点沿入边反向走，若走不到起点就会重复经过某个节点，形成环；删掉起点后余图仍无环，"
                "归纳即可排完。实现用 Kahn 算法逐个取入度为 $0$ 的节点；无环只能保证“排得出来”，不能保证"
                "顺序唯一，因此候选集按清单序号打破平局，输出才可复现。",
                "**分层取的是最长路径，不是章节号。** 定义 $\\operatorname{depth}(v)=0$（没有前置）或 "
                "$1+\\max_u\\operatorname{depth}(u)$（对所有 $u\\to v$）。由定义立刻得到 "
                "$\\operatorname{depth}(v)\\ge1+\\operatorname{depth}(u)>\\operatorname{depth}(u)$，"
                "所以每条边都从浅层指向更深一层，按深度分组就把节点分成了互不相交的层。最长链的长度恰为 "
                "$1+\\max_v\\operatorname{depth}(v)$：任何一条链每走一步深度至少加一，长度不会超过这个数；"
                "从最大深度节点沿“取到最大值的父节点”一路回溯，每步深度减一，又能构造出同样长的链。"
                "当前图有 9 层、最大深度 8、最宽一层 9 章，关键路径 9 章，最后落到 6.2。层号不是学期，"
                "也不是边的条数。",
                "**焦点闭包是它所有前置的传递闭包。** 对焦点章 $v$，从 $v$ 出发反复沿 deps 往回走，"
                "能到达的章连同 $v$ 自身组成祖先闭包 $C(v)$。它的含义很直接：$C(v)$ 里除 $v$ 以外的每一章"
                "都必须在 $v$ 之前构建，而 $v$ 之前的任何一次构建又必须包含它自己的全部前置。默认焦点 4.3 "
                "的闭包连同自身是 8 章，闭包内部有 9 条边；最长关键路径则穿过 1.1 到 6.2。读图时红圈只标"
                "焦点章本身，蓝色高亮的是闭包里的节点和两端都在闭包内的边；未选中的节点仍以灰色画出，"
                "因为一旦删掉它们，跨分部的依赖就会从图上消失。",
                "**层内顺序只是排版。** 深度只决定节点在第几列，同一层里谁上谁下并不影响依赖关系。"
                "为了让边少交叉，实现做三轮重心排序：先按父节点的平均纵向位置排，再按子节点的平均位置排，"
                "三轮后仍相同的按清单序号决定。这个纵向位置等于父节点位置的平均值："
                "$\\operatorname{pos}(v)=\\frac{1}{|\\mathcal{N}(v)|}\\sum_{u\\to v}\\operatorname{pos}(u)$。"
                "排序只改显示坐标，不改变边、拓扑序或深度。因此主图的横轴是深度，纵轴“层内序号”只是把"
                "同一列的节点摊开；同一份清单每次渲染都应得到同一张图，不能依赖集合的遍历顺序。",
                "**高亮不是筛选，参数要配合读。** 默认 part=all、focus=4.3 时，高亮完全由焦点闭包决定；"
                "把分部切到 IV 后，节点颜色改为该部的成员，边只要有一端属于该部就高亮，而焦点红圈与闭包"
                "章数仍按 4.3 计算——此时角标里的“边数”统计的是当前高亮边，不再专指闭包内部边。若把 focus "
                "设为 all，闭包变成整张图；若同时把 part 设为 all，则全部 28 个节点和 67 条边都高亮。"
                "副图的四格依次是环、未知依赖、自依赖、有清单但缺模块，全部为 $0$ 才说明依赖图的结构检查"
                "通过；它不检查正文、引用或编译，别把“四格全零”当成“整本书没问题”。",
            ],
        ),
        _viz(
            id="export_timeline",
            title="视频导出：帧数、时长与两本时间账",
            description=(
                "源时间轴来自自适应 dopri5（非均匀步长）。把 logistic 轨迹按固定 fps "
                "重采样：主图是源步长与导出帧的位置，副图是内容时刻误差（最近邻/线性/"
                "序号抽稀）与 GIF 厘秒量化造成的播放漂移。侧栏把名义帧数、实际帧数、"
                "截断、stall、状态误差与 delay_cs 分开报。"
            ),
            dataSource="backend",
            params=[
                {
                    "name": "tol",
                    "label": "自适应容差 tol",
                    "type": "enum",
                    "options": [1e-3, 1e-5, 1e-7],
                    "default": 1e-5,
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "自适应求解器每一步允许的局部误差容差，无量纲，档位是 1e-3、1e-5、1e-7。"
                        "调小（如 1e-7）会让求解器缩小步长、增加源快照，主图灰线上的台阶更密，"
                        "副图的内容时刻误差和侧栏的状态误差都下降；调大则步长变大、两种误差都变大。"
                        "它不影响 GIF 的播放漂移，因为漂移只由帧率对应的整数厘秒延迟决定。"
                        "先看侧栏的接受点数和步长范围，再看误差副图是否随档位改变。"
                    ),
                },
                {
                    "name": "fps",
                    "label": "请求帧率 fps",
                    "type": "enum",
                    "options": [12, 24, 25, 30, 60],
                    "default": 24,
                    "critical": [25.0],
                    "criticalNote": "25 fps 是 GIF 厘秒量化能精确表示的一档（4 cs）；24/30/60 都会被改写。",
                    "help": (
                        "请求的导出帧率，单位帧每秒，可选 12、24、25、30、60。调大后同样的 4 秒跨度会"
                        "产生更多帧，主图蓝点更密，也更容易撞上帧数上限而被截断；GIF 会把每帧间隔四舍五入"
                        "成整数厘秒，只有 25 fps 能精确表示，24 fps 实际按 25 fps 播放，30 与 60 也会被改写。"
                        "读侧栏的实际帧率、延迟厘秒数和片尾漂移，并对照副图橙虚线的斜率。"
                        "它和帧数上限一起决定最终帧数，和取样方式一起决定内容误差与卡顿。"
                    ),
                },
                {
                    "name": "sampling",
                    "label": "取样方式",
                    "type": "enum",
                    "options": ["nearest", "linear", "decimate"],
                    "default": "nearest",
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "每帧内容如何从源快照取得：最近邻、按物理时间线性插值，或按源序号等距抽稀。"
                        "最近邻把每帧贴到最近的源时刻，内容误差有半个源步长的上界，但同一快照会连续显示"
                        "多帧（主图红圈 stall）；线性插值让内容时刻精确、误差副图蓝线贴零，状态误差仍有二阶项；"
                        "序号抽稀看似首末对齐，在非均匀步长下会把物理时间拉弯。读副图蓝线和主图红圈，"
                        "并配合容差看源步长是否均匀。"
                    ),
                },
                {
                    "name": "budget",
                    "label": "帧数上限",
                    "type": "enum",
                    "options": [120, 240],
                    "default": 240,
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "允许导出的最大帧数，单位帧，档位是 120 与 240。名义帧数超过它时只保留前面的帧，"
                        "导出的时间跨度会变短；侧栏会显示已截断以及丢掉多少帧，主图与误差副图也提前结束。"
                        "调大上限可以让高帧率或长跨度完整导出，但如果本来没有超过上限，改它不会改变任何曲线。"
                        "它和请求帧率联动：60 fps 在 4 秒跨度需要 241 帧，240 的上限就会截掉最后一帧。"
                    ),
                },
            ],
            time={"mode": "none"},
            requiredAux=["error"],
            axes=[
                {
                    "id": "main",
                    "autoscaled": False,
                    "scale": "log",
                    "title": "自适应步长 $h(t)$ 与导出帧",
                    "ylabel": "步长 $h$（对数）",
                },
                {
                    "id": "error",
                    "autoscaled": False,
                    "title": "时间误差：内容时刻与 GIF 播放时刻（$\\Delta t$）",
                    "ylabel": "时间误差 $\\Delta t$（$s$）",
                },
            ],
            pitfalls=_PITFALLS["export_timeline"],
            theory=[
                "**源时间轴是自适应走出来的。** 导出页的源轨迹来自 logistic 方程 $y'=y(1-y)$、"
                "$y(0)=1/2$、$0\\le t\\le4$ 的 dopri5 积分。求解器每一步估计局部误差，只接受满足容差 tol "
                "的步长，于是接受点 $t_i$ 的间距 $h_i=t_{i+1}-t_i$ 天然不均匀；默认 tol=$10^{-5}$ 时只有 "
                "8 个接受点、7 步，$h$ 在 $0.05$ 到 $0.8444$ 之间，最大与最小之比约 $16.9$。主图灰线画的是"
                "每个接受步的步长，蓝点画导出帧，横坐标是名义播放时刻、纵坐标是该帧所用源步的 $h$"
                "（对数轴）。别把蓝点的横坐标当成内容时刻：内容时刻由取样方式另算，正是下面两本账的第一本。",
                "**名义帧数由跨度和帧率决定。** 设源跨度 $T=t_{\\mathrm{last}}-t_0>0$，请求帧率 $f>0$，"
                "第 $j$ 帧的名义时刻取 $\\tau_j=t_0+j/f$。为了不越过源末端，必须有 "
                "$\\tau_{n-1}\\le t_{\\mathrm{last}}$，也就是 $n-1\\le fT$；而 $n+1$ 帧时 "
                "$\\tau_n=t_0+n/f>t_{\\mathrm{last}}$ 已经越界。因此合法帧数就是满足 $n-1\\le fT<n$ 的"
                "那个整数 $n$，默认 $T=4$、$f=24$ 得 $n=97$。若帧数上限为 $K$，实际只导出 $\\min(n,K)$ "
                "帧，少掉的 $n-K$ 帧必须在侧栏标成“截断”，因为它缩短的是时间跨度，不是“数值只算到这里”。"
                "$f=60$ 时名义 241 帧，上限 240 会截掉最后一帧；$f=30$ 的 121 帧遇到上限 120 也一样。",
                "**最近邻把内容时刻量化到半个步长。** 对每个名义时刻 $\\tau_j$，最近邻在源时刻里找距离"
                "最小的 $t_i$，并让这一帧的内容时刻等于 $t_i$。源时刻把区间切成宽度不超过 $h_{\\max}$ 的"
                "小格，$\\tau_j$ 必落在某一格里，到最近端点的距离不超过该格宽度的一半，因此 "
                "$|t_i-\\tau_j|\\le h_i/2\\le h_{\\max}/2$。默认 $h_{\\max}=0.8444$ 给出上界 $0.4222$ s，"
                "实测最大误差 $0.4184$ s，略小是因为没有帧恰好落在最坏的中点；代码在中点时取靠前的快照。"
                "这个上界只保证“时间不太偏”，不保证画面平滑：源步长大于帧间隔 $1/f$ 时，连续多帧会取到"
                "同一个快照，形成台阶。默认 97 帧里有 89 帧与前一帧同源，最长连续保持 $0.7917$ s，"
                "这是独立的 stall 账。",
                "**线性插值让内容时刻精确，但状态仍有二阶误差。** 当 $\\tau$ 落在 $[t_i,t_{i+1}]$ 内，"
                "令权重 $w=(\\tau-t_i)/h_i$，取 $L_h y(\\tau)=(1-w)y_i+w y_{i+1}$。因为内容时刻直接取 "
                "$\\tau$ 本身，副图蓝线在 linear 模式下贴零，这不是“误差被藏起来”，而是取时刻这件事真的"
                "做准了。状态误差来自用直线代替曲线：若 $y$ 二阶连续可导，插值误差可写成 "
                "$\\frac{(\\tau-t_i)(\\tau-t_{i+1})}{2}y''(\\eta)$，两个因子的绝对值乘积不超过 "
                "$h_i^2/4$，于是 $|y-L_h y|\\le h_i^2\\max|y''|/8$。logistic 的 $|y''|$ 最大约 "
                "$0.0962$，系数约 $0.0121$；默认 $h_{\\max}$ 给出 $8.63\\times10^{-3}$ 的界，实测 "
                "$8.39\\times10^{-3}$，比最近邻的状态误差小一个数量级。注意 $h$ 由求解器容差决定，"
                "不由帧率决定：把 fps 调得再高，这个状态误差也不会自动消失；linear 模式下权重一直在变，"
                "所以 stall 计为 0。",
                "**按序号抽稀会把物理时间拉弯。** decimate 不看 $t$，而是把源下标从 $0$ 到 $n_s-1$ "
                "等距取 $n$ 个，第 $k$ 帧用下标 $\\operatorname{round}(k(n_s-1)/(n-1))$。这样首帧必是"
                "源起点、末帧必是源终点、下标严格不减，看起来“结构完全正确”；但相等的下标步长对应的"
                "时间步长是 $h_i$，在自适应网格上并不相等。默认 24 fps 时，最近邻的内容最大误差是 "
                "$0.4184$ s，抽稀却达到 $1.1167$ s，约为前者的 $2.7$ 倍；状态误差也从 "
                "$8.63\\times10^{-2}$ 升到 $2.30\\times10^{-1}$。在 9 点非均匀测试轴上更直观：最近邻 "
                "$0.5$ s，抽稀 $1.55$ s，是 $3.1$ 倍。这个误差没有 $h/2$ 之类的上界，步长比越大越糟；"
                "只有所有 $h_i$ 相等时它才退化成近似均匀取样。看到“首末对齐、下标单调”就相信时间轴，"
                "是这张图最想拆穿的误读。",
                "**GIF 的厘秒延迟把播放时刻改成另一本账。** GIF 每帧的 delay 只能存整数个厘秒，导出时取 "
                "$d=\\max(1,\\operatorname{round}(100/f))$，于是实际播放帧率是 $100/d$，而不是请求的 $f$。"
                "第 $j$ 帧的播放时刻是 $t_0+jd/100$，名义时刻是 $t_0+j/f$，两者相减得到累计漂移 "
                "$\\Delta_j=j(d/100-1/f)$：它随帧号线性增长，不是随机抖动。$f=24$ 时 $d=4$，实际 25 fps，"
                "片尾 $j=96$ 漂移 $-0.16$ s（播放偏快）；$f=25$ 时 $100/25=4$ 恰好是整数，漂移恒为 $0$；"
                "$f=12$ 也是 $-0.16$ s，$f=30$ 是 $-0.40$ s，$f=60$ 写成 2 厘秒、实际只有 50 fps，片尾"
                "漂移 $+0.80$ s（播放偏慢）。所以“请求 24 fps”不能当成“视频就是 24 fps”；只有能整除 100 "
                "的帧率才没有这项误差，而它和内容误差完全是两本账。侧栏的播放总时长按帧数乘 delay 计算，"
                "末帧也占满一个 delay，因此不等于最后一张的播放时刻。",
                "**两条曲线要分开读，调参也要分开调。** 主图看源步长和帧的位置：蓝点越密表示帧率越高，"
                "红圈表示这一帧与前帧同源。副图蓝线是内容时刻误差 $t_i-\\tau_j$（linear 下为 $0$），"
                "橙虚线是播放漂移 $\\Delta_j$；前者在最近邻下呈有界锯齿、在 linear 下贴零、在 decimate 下"
                "出现大跳变，后者总是一条过原点的直线。调小 tol 会让源点变密、$h$ 变小，因而蓝线和状态"
                "误差都下降，却完全不改橙线；把 fps 切到 25 会让橙线归零，却不会自动减小源步长带来的状态"
                "误差；linear 消掉内容误差和 stall，但仍有 $O(h^2)$ 的状态误差。帧数上限只负责截断，"
                "不改变每一帧的取样规则。读图时先问“我在看哪本账”，再问“这个参数动的是哪一本”，"
                "就不会把“高帧率”误当成“高精度”。",
            ],
        ),
        _viz(
            id="book_build",
            title="讲义构建：引用、附表与字形审计",
            description=(
                "逐章检查六件事：notes 存在且含定理、asserts_table 行数与断言报告的条数一致、"
                "引用在章内闭合、LaTeX 的 input 目标存在、notes+附表无 xelatex 无字形"
                "风险字符、pitfalls $\\ge800$ 字。热图行=章、列=检查项；副图是每章的问题数。"
            ),
            dataSource="backend",
            params=[
                {
                    "name": "part",
                    "label": "分部筛选",
                    "type": "enum",
                    "options": ["all", "I", "II", "III", "IV", "V", "VI"],
                    "default": "all",
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "按分部筛选热图里显示哪些章（I 到 VI，all 为全部 28 章）。切到某一部后，"
                        "行数、行标签、副图点数以及侧栏的合计问题数都只统计该部，未选中的章不参与；"
                        "它不改变每章六格的取值，也不改变每一格所在的列。读图时先确认当前分部，"
                        "再比较不同部的失败列分布。它和排序一起决定行序：筛选先缩小集合，排序再在这个集合内重排。"
                    ),
                },
                {
                    "name": "sort",
                    "label": "排序",
                    "type": "enum",
                    "options": ["catalog", "topo", "fail"],
                    "default": "topo",
                    "critical": [],
                    "criticalNote": "",
                    "help": (
                        "决定热图行与副图横轴章的排列方式：目录顺序、拓扑构建顺序，或问题优先。"
                        "选问题优先时，问题数多的章排在最上面，同分再按目录顺序；选拓扑顺序时，"
                        "行序服从依赖图的构建先后，不再等于章节号。它只改显示顺序，不改任何一格的通过或失败。"
                        "找失败章时先切到问题优先，再结合分部筛选缩小范围；读行标签与副图红点即可。"
                    ),
                },
            ],
            time={"mode": "none"},
            requiredAux=["error"],
            axes=[
                {
                    "id": "main",
                    "autoscaled": False,
                    "title": "逐章构建检查（$1$=通过，$0$=缺）",
                    "ylabel": "章（行 $0$ 在底部）",
                },
                {
                    "id": "error",
                    "autoscaled": False,
                    "title": "每章问题数（$0$=通过；下留白 $0.5$ 仅为显示 $0$ 点）",
                    "ylabel": "问题数（$0$ 到 $6$）",
                },
            ],
            pitfalls=_PITFALLS["book_build"],
            theory=[
                "**每章先变成一个六位布尔向量。** 对第 $i$ 章，检查六件事并记 $v_{ij}$ 为 $0$ 或 $1$："
                "notes.tex 存在且至少含一个定理环境；asserts_table.tex 存在且正文行数等于断言报告的条数；"
                "notes 里的引用都能在本章标签中找到，且没有重复标签；LaTeX 的 input 目标文件都存在；"
                "notes 与附表（剔除注释后）没有缺字形风险字符；pitfalls.md 至少 800 字。热图的行是章、"
                "列是六项检查，亮黄色格表示 $1$、深紫色格表示 $0$；副图把每章的问题数画成 "
                "$6-\\sum_{j=1}^{6}v_{ij}$。28 章乘 6 项就是 168 个格子，任何一个 0 都不能被同章其他 1 "
                "抵消——这里没有加权平均，只有逐格对账。别把“六格全绿”理解成 PDF 一定好看，它只说明这六条"
                "可自动检查的结构条件成立。",
                "**引用必须在本章内闭合。** LaTeX 的标签和引用构成两个集合：本章所有 label 名组成 $L$，"
                "所有 ref 与 eqref 的目标组成 $R$。检查要求 $R$ 中每个名字都能在 $L$ 里找到，并且 $L$ 中"
                "没有重复项；注释里出现的命令不算，因为解析器先按未转义的百分号剔除注释。为什么不能靠合并本？"
                "因为讲义是逐章单独编译的：合并本里另一个章的 label 能解析，单章 wrapper 里它就是未定义"
                "引用，编译只会给出问号或警告，不会告诉你缺了哪一章。当前 28 章统计到 539 个标签、445 个"
                "引用、28 个 input，未定义引用和重复标签都是 0。注意“编译两遍能消掉问号”只对同一文档内"
                "已定义的标签成立，跨章引用不会因为多编译一遍就出现。",
                "**附表对的是报告条数，不是测试函数数。** 断言附表由同一份断言报告生成：报告里每登记"
                "一条断言记录，附表就多一行；而 asserts.yaml 声明的是测试函数，一个函数可以登记多条记录。"
                "所以正确的等式是“附表正文行数 = 报告里的断言条数”，写成 "
                "$r_{\\mathrm{table}}=r_{\\mathrm{report}}$；拿它去比测试函数数会得到假的不一致。"
                "仓库里 1.1 章的 27 个测试函数产生 31 条记录，1.5 章的 24 个函数产生 71 条记录，就是现成"
                "的反例。当前 28 章合计 3201 行、报告 3201 条。若某章报告缺失或没有“断言条数”字段，检查"
                "记 0 而不是猜一个数；附表少一行也会让这一格失败，因为纸面数字必须与机器记录同源。",
                "**字形检查针对的是静默失败。** xelatex 遇到字体里没有的码位时，通常只在日志里打印 "
                "Missing character，然后继续编译，退出码仍是 0，PDF 里那个字就消失了；所以“编译通过”不能"
                "证明字都在。检查把 notes 和附表先去掉注释，再按白名单扫描：ASCII、常用汉字、希腊字母和"
                "已验证的数学符号之外，任何字符都记为风险。它必须在生成 PDF 之前做，不能等看日志，因为日志"
                "可能被截断或淹没。当前 28 章扫描结果是 0 个风险字符；这个数字只说明当前白名单覆盖了正文"
                "用到的字符，不说明字体在所有机器上都一样。常见误读是把注释里的怪字符也算进去——注释不会"
                "进 PDF，先剔除再数才公平。",
                "**行序是显示契约，不是数据本身。** 热图约定第 0 行画在纵轴最小值处，而读者希望目录第一"
                "的章显示在最上面，所以模型把数据行翻转：显示的第 $i$ 行对应数组的第 $n-1-i$ 行。行的排列"
                "由排序参数决定：目录顺序、拓扑构建顺序，或问题优先；问题优先按每章问题数从多到少排，"
                "同分再按目录顺序，所以失败章会被顶到前面。分部筛选先缩小章集合，再在集合内排序，因此行数、"
                "副图点数和侧栏合计都只针对当前筛选范围，切换分部时合计会变；排序只改顺序，不会改任何一格"
                "的 0/1。读图时先看行标签确认是哪一章，再看列标签确认缺的是哪一项，最后看副图红点定位问题数，"
                "不要拿颜色深浅当连续数值——这里只有通过和不通过两种状态。",
                "**每次重扫磁盘，结果才可复现。** 这个页面不读取上一次构建留下的报告缓存，而是对当前仓库"
                "逐章读取 notes、附表、报告、pitfalls，重新算六位向量；同一份源文件在同一版本下得到同一组"
                "格子，这就是可复现构建的含义。把每章的六位结果按固定章序拼起来，可以得到一个构建指纹："
                "任何一格从 1 变 0，指纹就变，适合用来判断“这次提交到底改坏了什么”。但指纹相同只说明六项"
                "检查相同，不说明正文内容没变；反过来，编译成功也不能替代指纹，因为缺字形和跨章引用都可能"
                "静默通过。当前账本是 168 格全 1、问题 0、定理 284、标签 539、引用 445、input 28、附表 "
                "3201 行、风险字符 0；这些数字会随章节增删变化，所以页面每次都重算，而不是相信旧结论。",
            ],
        ),
    ],
}


# --------------------------------------------------------------------------- #
# 公共工具
# --------------------------------------------------------------------------- #
def _finite(v: Any, fallback: float = 0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return fallback
    return x if math.isfinite(x) else fallback


def _cid_key(cid: str) -> tuple[int, int]:
    a, b = cid.split(".")
    return int(a), int(b)


def _json_list(values: Any) -> str:
    return json.dumps([str(v) for v in values], ensure_ascii=False, separators=(",", ":"))


# --------------------------------------------------------------------------- #
# 1. 目录依赖图
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def _catalog_graph() -> dict[str, Any]:
    from server.registry import load_chapter_yaml

    catalog = load_chapter_yaml(_ROOT)
    ids = sorted(catalog, key=_cid_key)
    raw_deps = {cid: list((catalog[cid].get("deps") or [])) for cid in ids}
    deps, dep_problems = packaging.resolve_dependencies(ids, raw_deps)
    order, cycles = packaging.topological_order(ids, deps)
    if cycles:
        raise ValueError(f"docs/chapters.yaml 的依赖图有环：{cycles}")
    depth = packaging.dependency_depth(ids, deps)
    chain = packaging.longest_chain(ids, deps)
    layers = [list(layer) for layer in packaging.dependency_layers(ids, deps)]
    rank = {cid: i for i, cid in enumerate(ids)}
    parents = {cid: list(deps[cid]) for cid in ids}
    children: dict[str, list[str]] = {cid: [] for cid in ids}
    for cid in ids:
        for dep in deps[cid]:
            children[dep].append(cid)
    # 层内用重心法排一遍（上行看父节点、下行看子节点），减少交叉；同重心按清单序。
    ypos: dict[str, float] = {}
    for d, layer in enumerate(layers):
        m = len(layer)
        for i, cid in enumerate(layer):
            ypos[cid] = float(i) - 0.5 * (m - 1)

    def _bary(cid: str, refs: Mapping[str, Sequence[str]], sign: float) -> float:
        vals = [sign * ypos[r] for r in refs.get(cid, ()) if r in ypos]
        return float(sum(vals) / len(vals)) if vals else ypos.get(cid, 0.0)

    for _sweep in range(3):
        for d in range(1, len(layers)):
            layers[d].sort(key=lambda c: (_bary(c, parents, 1.0), rank[c]))
            m = len(layers[d])
            for i, cid in enumerate(layers[d]):
                ypos[cid] = float(i) - 0.5 * (m - 1)
        for d in range(len(layers) - 2, -1, -1):
            layers[d].sort(key=lambda c: (_bary(c, children, 1.0), rank[c]))
            m = len(layers[d])
            for i, cid in enumerate(layers[d]):
                ypos[cid] = float(i) - 0.5 * (m - 1)
    pos = {cid: (float(depth[cid]), ypos[cid]) for cid in ids}
    edges: list[tuple[str, str]] = []
    for cid in ids:
        for dep in deps[cid]:
            edges.append((dep, cid))
    parts = {cid: str(catalog[cid].get("part", "")) for cid in ids}
    vizzes = {cid: int(catalog[cid].get("viz", 0) or 0) for cid in ids}
    titles = {cid: str(catalog[cid].get("title", "")) for cid in ids}
    return {
        "ids": ids,
        "deps": deps,
        "order": order,
        "depth": depth,
        "layers": layers,
        "chain": chain,
        "edges": edges,
        "pos": pos,
        "parts": parts,
        "vizzes": vizzes,
        "titles": titles,
        "problems": dep_problems,
    }


def _ancestor_closure(focus: str, deps: Mapping[str, Sequence[str]]) -> set[str]:
    if focus == "all":
        return set(deps)
    seen: set[str] = set()
    stack = [focus]
    while stack:
        cid = stack.pop()
        if cid in seen or cid not in deps:
            continue
        seen.add(cid)
        stack.extend(deps[cid])
    return seen


def _catalog_nav(params: Mapping[str, Any]) -> Frame:
    g = _catalog_graph()
    ids: list[str] = g["ids"]
    deps = g["deps"]
    pos = g["pos"]
    part = str(params.get("part", "all"))
    focus = str(params.get("focus", "4.3"))
    if part not in ("all", *_PARTS):
        raise ValueError(f"未知分部 {part!r}")
    if focus != "all" and focus not in pos:
        raise ValueError(f"未知焦点章 {focus!r}")
    closure = _ancestor_closure(focus, deps)
    part_mask = np.asarray(
        [1.0 if part == "all" or g["parts"][cid] == part else 0.0 for cid in ids], dtype=float
    )
    focus_mask = np.asarray([1.0 if cid in closure else 0.0 for cid in ids], dtype=float)
    node_hl = np.where(part == "all", focus_mask, part_mask)
    is_focus = np.asarray([1.0 if cid == focus else 0.0 for cid in ids], dtype=float)
    edge_hl = np.asarray(
        [
            1.0 if (a in closure and b in closure and part == "all") or (part != "all" and (g["parts"].get(a) == part or g["parts"].get(b) == part)) else 0.0
            for a, b in g["edges"]
        ],
        dtype=float,
    )
    if focus == "all" and part == "all":
        node_hl = np.ones(len(ids), dtype=float)
        edge_hl = np.ones(len(g["edges"]), dtype=float)

    node_x = np.asarray([pos[cid][0] for cid in ids], dtype=float)
    node_y = np.asarray([pos[cid][1] for cid in ids], dtype=float)
    node_depth = np.asarray([g["depth"][cid] for cid in ids], dtype=float)
    node_part = np.asarray([float(_ROMAN.get(g["parts"][cid], 0)) for cid in ids], dtype=float)
    node_viz = np.asarray([float(g["vizzes"][cid]) for cid in ids], dtype=float)
    node_rank = np.zeros(len(ids), dtype=float)
    for rank, cid in enumerate(g["order"]):
        node_rank[ids.index(cid)] = float(rank)
    node_deps = np.asarray([float(len(deps[cid])) for cid in ids], dtype=float)
    e = g["edges"]
    edge_x0 = np.asarray([pos[a][0] for a, _ in e], dtype=float) if e else np.zeros(0)
    edge_y0 = np.asarray([pos[a][1] for a, _ in e], dtype=float) if e else np.zeros(0)
    edge_x1 = np.asarray([pos[b][0] for _, b in e], dtype=float) if e else np.zeros(0)
    edge_y1 = np.asarray([pos[b][1] for _, b in e], dtype=float) if e else np.zeros(0)

    n_unknown = sum(1 for p in g["problems"] if "未知依赖" in p)
    n_self = sum(1 for p in g["problems"] if "自依赖" in p)
    # 第四格"目录漂移"：清单里的 id 是否都能在 MANIFEST 生成的目录中找到。
    # 这里不读 dist；用服务器已实现的章节文件数量与清单数量的差作为构建期缺口。
    module_ids = {
        p.stem
        for p in (_ROOT / "server" / "chapters").glob("*.py")
        if re.fullmatch(r"\d+\.\d+", p.stem)
    }
    n_missing_module = sum(1 for cid in ids if cid not in module_ids)
    error_values = np.asarray([0.0, float(n_unknown), float(n_self), float(n_missing_module)], dtype=float)

    fr = Frame(chapter=CHAPTER, viz="catalog_nav", kind="frame")
    fr.add("node_x", node_x, label="节点依赖层")
    fr.add("node_y", node_y, label="层内序号")
    fr.add("node_depth", node_depth, label="拓扑深度")
    fr.add("node_part", node_part, label="分部编号（I..VI）")
    fr.add("node_viz", node_viz, label="可视化数")
    fr.add("node_rank", node_rank, label="拓扑序")
    fr.add("node_deps", node_deps, label="直接前置数")
    fr.add("node_highlight", node_hl, label="高亮掩码")
    fr.add("node_focus", is_focus, label="焦点章掩码")
    fr.add("edge_x0", edge_x0, label="边起点 x")
    fr.add("edge_y0", edge_y0, label="边起点 y")
    fr.add("edge_x1", edge_x1, label="边终点 x")
    fr.add("edge_y1", edge_y1, label="边终点 y")
    fr.add("edge_highlight", edge_hl, label="边高亮掩码")
    fr.add("error", error_values, label="依赖图错误计数（requiredAux）")
    fr.add("error_values", error_values, label="错误计数")
    fr.add("layer_size", np.asarray([float(len(layer)) for layer in g["layers"]]), label="每层节点数")
    fr.set_meta(
        n_chapters=len(ids),
        n_edges=len(e),
        n_layers=len(g["layers"]),
        max_depth=float(max(g["depth"].values(), default=0)),
        n_cycles=0.0,
        n_unknown=float(n_unknown),
        n_self=float(n_self),
        n_missing_module=float(n_missing_module),
        n_violations=float(sum(error_values)),
        focus_closure=float(len(closure)) if focus != "all" else float(len(ids)),
        focus_edges=float(sum(edge_hl)) if focus != "all" else float(len(e)),
        chain_len=float(len(g["chain"])),
        overflow=False,
    )
    fr.set_strings(
        status="ok" if sum(error_values) == 0 else "degraded",
        chapter_ids=_json_list(ids),
        parts=_json_list([g["parts"][cid] for cid in ids]),
        titles=_json_list([g["titles"][cid] for cid in ids]),
        focus=focus,
        part=part,
        chain=" → ".join(g["chain"]),
        problem_note=(
            "依赖图四项检查全为 0：无环、无未知依赖、无自依赖、清单章都有模块。"
            if sum(error_values) == 0
            else "依赖图有问题：" + "；".join(g["problems"] + [f"缺模块 {n_missing_module} 章"])
        ),
        closure_note=(
            "焦点=all：显示全图。"
            if focus == "all"
            else f"焦点 {focus} 的前置闭包含 {len(closure)} 章；这些章必须在它之前构建。"
        ),
        layer_note=f"{len(g['layers'])} 层、{len(e)} 条边；最长链 {len(g['chain'])} 章（关键路径）。",
        error_note="四格：环 / 未知依赖 / 自依赖 / 有清单但缺模块。",
    )
    fr.pitfalls = _PITFALLS["catalog_nav"]
    return fr


# --------------------------------------------------------------------------- #
# 2. 视频导出：内容时刻与播放时刻
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=12)
def _adaptive_source(tol: float) -> tuple[np.ndarray, np.ndarray, Any]:
    p = rk.problem("logistic")
    run = integrate_adaptive(
        "dopri5",
        p.rhs,
        (0.0, 4.0),
        p.y0_array(),
        float(tol),
        atol=float(tol),
        h0=0.05,
        local_extrapolation=True,
        exact=p.exact,
        label="logistic",
    )
    if not run.ok or run.t.size < 2:
        raise RuntimeError("自适应 logistic 积分失败，无法构造导出时间轴")
    return np.asarray(run.t, dtype=float), np.asarray(run.y, dtype=float), run


def _export_timeline(params: Mapping[str, Any]) -> Frame:
    tol = float(params["tol"])
    fps = float(params["fps"])
    mode = str(params["sampling"])
    budget = int(params["budget"])
    if tol not in (1e-3, 1e-5, 1e-7):
        raise ValueError(f"未知容差 {tol!r}")
    if fps not in (12.0, 24.0, 25.0, 30.0, 60.0):
        raise ValueError(f"未知 fps {fps!r}")
    if mode not in ("nearest", "linear", "decimate"):
        raise ValueError(f"未知取样方式 {mode!r}")
    if budget not in (120, 240):
        raise ValueError(f"未知帧数上限 {budget!r}")
    tt, yy, run = _adaptive_source(tol)
    plan = packaging.export_plan(
        tt,
        yy,
        fps,
        mode=mode,
        max_frames=budget,
        exact=rk.problem("logistic").exact,
    )
    fr = Frame(chapter=CHAPTER, viz="export_timeline", kind="frame")
    fr.add("source_t", plan.source_t, label="自适应接受步时刻")
    fr.add("source_step_t", plan.source_step_t, label="步长起点时刻")
    fr.add("source_step_h", plan.source_step_h, label="自适应步长 h")
    fr.add("frame_t", plan.frame_t, label="名义帧时刻")
    fr.add("frame_index", np.arange(plan.n_frames, dtype=float), label="帧号")
    fr.add("content_t", plan.content_t, label="内容时刻")
    fr.add("playback_t", plan.playback_t, label="GIF 播放时刻")
    fr.add("content_index", plan.content_index.astype(float), label="源快照下标")
    fr.add("weight", plan.weight, label="线性插值权重")
    fr.add("content_y", plan.content_y[:, 0], label="帧内容 y")
    fr.add("exact_y", plan.exact_y[:, 0] if plan.exact_y is not None else np.zeros(plan.n_frames), label="解析解 y")
    fr.add("error", plan.timing_error, label="内容时刻误差（requiredAux）")
    fr.add("timing_error", plan.timing_error, label="内容时刻误差")
    fr.add("playback_error", plan.playback_error, label="GIF 播放漂移")
    fr.add("state_error", plan.state_error, label="帧内容相对解析解的误差")
    fr.add("stall", plan.stall.astype(float), label="同源快照重复掩码")
    fr.set_meta(
        tol=float(tol),
        fps=float(fps),
        delay_cs=float(plan.delay_cs),
        actual_fps=float(plan.actual_fps),
        fps_error=float(plan.actual_fps - fps),
        n_source=int(plan.source_t.size),
        n_steps=int(run.n_accepted),
        n_rejected=int(run.n_rejected),
        h_min=float(np.min(plan.source_step_h)),
        h_max=float(np.max(plan.source_step_h)),
        h_ratio=float(np.max(plan.source_step_h) / np.min(plan.source_step_h)),
        span=float(plan.span),
        n_raw=int(plan.n_raw),
        n_frames=int(plan.n_frames),
        truncated=1.0 if plan.truncated else 0.0,
        n_truncated=float(plan.n_raw - plan.n_frames),
        nominal_duration=float(plan.nominal_duration),
        playback_duration=float(plan.playback_duration),
        duration_gap=float(plan.duration_gap),
        max_timing_error=float(plan.max_abs_timing_error),
        max_playback_error=float(plan.max_abs_playback_error),
        max_state_error=float(plan.max_state_error),
        n_stalls=float(plan.n_stalls),
        max_hold=float(plan.max_hold),
        max_hold_playback=float(plan.max_hold_playback),
        n_skipped_source=float(plan.n_skipped_source),
        overflow=False,
    )
    mode_label = {
        "nearest": "最近邻快照",
        "linear": "物理时间线性插值",
        "decimate": "按序号等距抽稀",
    }[mode]
    fr.set_strings(
        status="ok",
        sampling=mode,
        sampling_note=f"取样={mode_label}；帧内容取自源快照，误差另算。",
        source_note=(
            f"自适应 dopri5：{plan.source_t.size} 个接受点、{run.n_rejected} 次拒绝；"
            f"h∈[{np.min(plan.source_step_h):.4g}, {np.max(plan.source_step_h):.4g}]，"
            f"最大/最小={np.max(plan.source_step_h)/np.min(plan.source_step_h):.3g}。"
        ),
        frame_note=(
            f"名义 {plan.n_raw} 帧，上限 {budget}，实出 {plan.n_frames} 帧"
            + (f"（截断 {plan.n_raw - plan.n_frames} 帧）" if plan.truncated else "（未截断）")
            + f"；stall {plan.n_stalls} 帧，最长同源 {plan.max_hold:.4f} s。"
        ),
        gif_note=(
            f"GIF delay={plan.delay_cs} cs（请求 {fps:g} fps → 实际 {plan.actual_fps:.4f} fps）；"
            f"播放总时长 {plan.playback_duration:.4f} s，名义 {plan.nominal_duration:.4f} s，"
            f"片尾漂移 {plan.max_abs_playback_error:.4f} s。"
        ),
        state_note=(
            f"最近邻/插值的帧内容相对解析解 max|Δy|={plan.max_state_error:.3e}"
            f"（源步长二阶项主导）。"
        ),
        error_note=(
            "内容误差有界（≤h/2，最近邻），播放漂移按 delay_cs 线性累积；"
            "decimate 模式把非均匀步长按序号拉平，误差不再有界。"
        ),
    )
    fr.pitfalls = _PITFALLS["export_timeline"]
    return fr


# --------------------------------------------------------------------------- #
# 3. 讲义构建审计
# --------------------------------------------------------------------------- #
def _strip_text(path: pathlib.Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _declared_assert_count(cid: str) -> int:
    import yaml

    path = _ROOT / "chapters" / cid / "asserts.yaml"
    if not path.exists():
        return 0
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return 0
    return len(data.get("tests") or [])


def _report_assert_count(cid: str) -> int:
    """``asserts_report.md`` 里的实测断言条数；没有报告返回 -1。

    附表由**同一份断言报告**生成（``tools/build_book.py``），所以"附表行数=报告条数"
    才是纸面与机器同源。拿它去对 ``asserts.yaml`` 的**测试函数数**会错：一个测试可以
    登记多条断言（1.1 章 27 个函数 → 31 条记录）。
    """
    path = _ROOT / "chapters" / cid / "asserts_report.md"
    text = _strip_text(path)
    if text is None:
        return -1
    m = re.search(r"断言条数：\s*(\d+)", text)
    return int(m.group(1)) if m else -1


def _resolve_input(target: str) -> pathlib.Path:
    rel = target.strip()
    p = _ROOT / rel
    if not p.suffix:
        p = _ROOT / (rel + ".tex")
    return p


@functools.lru_cache(maxsize=1)
def _book_audit_base() -> dict[str, Any]:
    from tools.build_book import latex_risky_chars

    g = _catalog_graph()
    ids: list[str] = g["ids"]
    rows: dict[str, dict[str, Any]] = {}
    for cid in ids:
        notes_path = _ROOT / "chapters" / cid / "notes.tex"
        table_path = _ROOT / "chapters" / cid / "asserts_table.tex"
        pitfalls_path = _ROOT / "chapters" / cid / "pitfalls.md"
        notes = _strip_text(notes_path)
        table = _strip_text(table_path)
        pitfalls = _strip_text(pitfalls_path)
        clean_notes = packaging.strip_tex_comments(notes or "")
        clean_table = packaging.strip_tex_comments(table or "")
        cross = packaging.tex_crossrefs(notes or "")
        labels = set(cross.labels)
        unresolved = [r for r in cross.refs if r not in labels]
        dup = cross.duplicate_labels
        inputs_missing = [t for t in cross.inputs if not _resolve_input(t).exists()]
        declared_tests = _declared_assert_count(cid)
        report_count = _report_assert_count(cid)
        table_rows = packaging.asserts_table_rows(table or "")
        risky = latex_risky_chars(clean_notes + "\n" + clean_table)
        checks = (
            bool(notes is not None and clean_notes.count("\\begin{theorem}") > 0),
            bool(table is not None and report_count > 0 and table_rows == report_count),
            bool(notes is not None and not unresolved and not dup),
            bool(notes is not None and not inputs_missing),
            bool(notes is not None and not risky),
            bool(pitfalls is not None and len(pitfalls.strip()) >= 800),
        )
        rows[cid] = {
            "checks": tuple(bool(x) for x in checks),
            "labels": len(cross.labels),
            "refs": len(cross.refs),
            "inputs": len(cross.inputs),
            "unresolved": len(unresolved),
            "duplicate": len(dup),
            "inputs_missing": len(inputs_missing),
            "risky": len(risky),
            "risky_chars": "".join(risky),
            "table_rows": table_rows,
            "declared_tests": declared_tests,
            "report_count": report_count,
            "theorems": clean_notes.count("\\begin{theorem}"),
            "chars": len(notes or ""),
            "pitfall_chars": len((pitfalls or "").strip()),
            "missing_inputs": tuple(inputs_missing),
        }
    return {"ids": ids, "rows": rows}


def _book_order(ids: list[str], sort: str) -> list[str]:
    g = _catalog_graph()
    if sort == "catalog":
        return list(ids)
    if sort == "topo":
        rank = {cid: i for i, cid in enumerate(g["order"])}
        return sorted(ids, key=lambda c: rank.get(c, 10**9))
    if sort == "fail":
        base = _book_audit_base()
        rank = {cid: i for i, cid in enumerate(ids)}
        return sorted(ids, key=lambda c: (-(len(_CHECK_IDS) - sum(base["rows"][c]["checks"])), rank[c]))
    raise ValueError(f"未知排序 {sort!r}")


def _book_build(params: Mapping[str, Any]) -> Frame:
    part = str(params.get("part", "all"))
    sort = str(params.get("sort", "topo"))
    if part not in ("all", *_PARTS):
        raise ValueError(f"未知分部 {part!r}")
    base = _book_audit_base()
    g = _catalog_graph()
    ids = [cid for cid in g["ids"] if part == "all" or g["parts"][cid] == part]
    ids = _book_order(ids, sort)
    rows = base["rows"]
    n = len(ids)
    values = np.zeros((n, len(_CHECK_IDS)), dtype=float)
    for i, cid in enumerate(ids):
        values[i, :] = np.asarray(rows[cid]["checks"], dtype=float)
    issues = np.asarray([len(_CHECK_IDS) - int(np.sum(values[i])) for i in range(n)], dtype=float)
    labels = np.asarray([float(rows[c]["labels"]) for c in ids], dtype=float)
    refs = np.asarray([float(rows[c]["refs"]) for c in ids], dtype=float)
    inputs = np.asarray([float(rows[c]["inputs"]) for c in ids], dtype=float)
    risky = np.asarray([float(rows[c]["risky"]) for c in ids], dtype=float)
    table_rows = np.asarray([float(rows[c]["table_rows"]) for c in ids], dtype=float)
    declared = np.asarray([float(rows[c]["report_count"]) for c in ids], dtype=float)
    declared_tests = np.asarray([float(rows[c]["declared_tests"]) for c in ids], dtype=float)
    theorems = np.asarray([float(rows[c]["theorems"]) for c in ids], dtype=float)
    chars = np.asarray([float(rows[c]["chars"]) for c in ids], dtype=float)

    fr = Frame(chapter=CHAPTER, viz="book_build", kind="frame")
    # 热图行 0 在 y-min（光栅器的约定）；行标签由前端按同一顺序翻转。
    fr.add("check_values", values.reshape(-1), label="逐章检查矩阵（行主序）")
    fr.add("chapter_index", np.arange(n, dtype=float), label="显示行号")
    fr.add("error", issues, label="每章问题数（requiredAux）")
    fr.add("error_y", issues, label="每章问题数")
    fr.add("labels", labels, label="LaTeX 标签命令数")
    fr.add("refs", refs, label="\\ref 数")
    fr.add("inputs", inputs, label="LaTeX 输入命令数")
    fr.add("risky", risky, label="字形风险字符数")
    fr.add("table_rows", table_rows, label="附表行数")
    fr.add("declared", declared, label="报告断言条数")
    fr.add("declared_tests", declared_tests, label="asserts.yaml 声明的测试函数数")
    fr.add("theorems", theorems, label="定理数")
    fr.add("chars", chars, label="正文字符数")
    total_labels = int(np.sum(labels))
    total_refs = int(np.sum(refs))
    total_inputs = int(np.sum(inputs))
    total_risky = int(np.sum(risky))
    total_issues = int(np.sum(issues))
    missing_tables = sum(1 for c in ids if rows[c]["table_rows"] == 0)
    mismatched = sum(1 for c in ids if rows[c]["table_rows"] != rows[c]["report_count"])
    unresolved = sum(int(rows[c]["unresolved"]) for c in ids)
    duplicates = sum(int(rows[c]["duplicate"]) for c in ids)
    missing_inputs = sum(int(rows[c]["inputs_missing"]) for c in ids)
    bad_rows = [
        f"{cid}: " + "、".join(label for label, ok in zip(_CHECK_LABELS, rows[cid]["checks"]) if not ok)
        for cid in ids
        if not all(rows[cid]["checks"])
    ]
    fr.set_meta(
        n_chapters=float(n),
        n_checks=float(len(_CHECK_IDS)),
        n_issues=float(total_issues),
        n_labels=float(total_labels),
        n_refs=float(total_refs),
        n_inputs=float(total_inputs),
        n_risky=float(total_risky),
        n_unresolved=float(unresolved),
        n_duplicates=float(duplicates),
        n_missing_inputs=float(missing_inputs),
        n_missing_tables=float(missing_tables),
        n_table_mismatch=float(mismatched),
        n_table_rows=float(np.sum(table_rows)),
        n_declared=float(np.sum(declared)),
        n_declared_tests=float(np.sum(declared_tests)),
        n_theorems=float(np.sum(theorems)),
        overflow=False,
    )
    fr.set_strings(
        status="ok" if total_issues == 0 else "degraded",
        chapter_ids=_json_list(ids),
        check_ids=_json_list(_CHECK_IDS),
        check_labels=_json_list(_CHECK_LABELS),
        part=part,
        sort=sort,
        summary=f"{n} 章 × {len(_CHECK_IDS)} 项检查；问题格 {total_issues} 个。",
        issue_note=("全部检查通过。" if not bad_rows else "未通过：" + "；".join(bad_rows[:6]) + ("…" if len(bad_rows) > 6 else "")),
        ref_note=(
            f"标签 {total_labels}、引用 {total_refs}、输入 {total_inputs}；"
            f"未定义 {unresolved}、重复标签 {duplicates}、缺输入文件 {missing_inputs}。"
        ),
        table_note=(
            f"附表行数 {int(np.sum(table_rows))}，报告条数 {int(np.sum(declared))}；"
            f"不一致 {mismatched} 章，缺表 {missing_tables} 章。"
        ),
        glyph_note=f"字形风险字符 {total_risky} 个（notes+附表，注释已剔除）。",
        pitfall_note=(
            "pitfalls 全部 ≥800 字。"
            if all(rows[c]["pitfall_chars"] >= 800 for c in ids)
            else "pitfalls 不足 800 字：" + "、".join(c for c in ids if rows[c]["pitfall_chars"] < 800)
        ),
        check_note="列：笔记存在且含定理 / 附表行数=声明 / 章内引用闭合 / LaTeX input 目标存在 / 无字形风险 / pitfalls≥800 字。",
    )
    fr.pitfalls = _PITFALLS["book_build"]
    return fr


# --------------------------------------------------------------------------- #
def compute(viz_id: str, params: Mapping[str, Any], t: float) -> Frame:
    """章节计算入口；三个页面都没有时间轴，``t`` 由调度器给 0。"""
    del t
    if viz_id == "catalog_nav":
        return _catalog_nav(params)
    if viz_id == "export_timeline":
        return _export_timeline(params)
    if viz_id == "book_build":
        return _book_build(params)
    raise KeyError(f"章节 6.2 没有可视化 {viz_id!r}")
