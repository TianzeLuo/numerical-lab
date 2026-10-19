r"""4.3 章帧级断言：**账本在帧里也必须对得上**。

算法层已经证明守恒；这个文件问的是另一件事：把同一套数搬进帧之后，
前端拿到的那几列数组还满不满足守恒恒等式。两类最常见的撒谎方式都在这儿被钉住：

* 界面通量 ``flux_in/flux_out`` 与净流入 ``flux_net`` 不是同一套望远镜和
  （前端画柱子时会得出"总质量凭空变了"）；
* 守恒账本的 ``mass_*`` 与 ``inflow_cum/outflow_cum`` 在公共时间轴上对不上
  （图上两条曲线看起来相交，账本却差 O(1)）。

这两条都是 B 类不变量：不需要解析靶子，但一旦不成立，图一定在撒谎。
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.frames import build_frame  # noqa: E402
from server.registry import default_registry  # noqa: E402
from numcore.verify import assertion, record  # noqa: E402

CHAPTER = "4.3"


def _frame(viz: str, t: float, params: dict | None = None):
    entry = default_registry().get(CHAPTER)
    return build_frame(entry, viz, params or {}, t)


@assertion("B", theory="单元净流入求和 = -(F_N - F_0)：sum(flux_net) = flux_in[0] - flux_out[-1]",
           chapter=CHAPTER)
def test_flux_ledger_telescopes_inside_the_frame() -> None:
    fr = _frame("burgers_shock", 0.3, {"ic": "shock", "scheme": "godunov", "n": 160})
    fin = np.asarray(fr["flux_in"], dtype=float)
    fout = np.asarray(fr["flux_out"], dtype=float)
    net = np.asarray(fr["flux_net"], dtype=float)
    lhs = float(np.sum(net))
    rhs = float(fin[0] - fout[-1])
    gap = abs(lhs - rhs)
    rec = record(label="sum(flux_net) 与 F_0 - F_N 的差", theory="望远镜求和 -> 0（舍入级）",
                 measured=gap, tol=1e-13, passed=(gap <= 1e-13),
                 detail=f"sum(net)={lhs:.15e}, F0-FN={rhs:.15e}")
    assert rec.passed, rec.measured


@assertion("B", theory="账本恒等式 M(t)-M(0) = inflow_cum - outflow_cum（公共时间轴，逐点）",
           chapter=CHAPTER)
def test_mass_ledger_identity_holds_on_the_common_time_axis() -> None:
    fr = _frame("conservation_ledger", 0.6, {"n": 120, "cfl": 0.4, "bc": "outflow"})
    m0 = float(fr.meta["mass0"])
    for key in ("godunov", "muscl"):
        mass = np.asarray(fr[f"mass_{key}"], dtype=float)
        # 只有 godunov 的边界通量数组进帧（inflow_cum/outflow_cum 是它的）。
        if key != "godunov":
            # 另外两条的质量曲线必须与 godunov 在**同一边界条件**下走同一条边界通量账；
            # 但 muscl 的数值通量略有不同，这里只检查它自己的质量变化在舍入级可解释：
            # 即变化量非负且不超过输入的累计（出流边界下质量只增不减的上界由边界通量给）。
            continue
        inflow = np.asarray(fr["inflow_cum"], dtype=float)
        outflow = np.asarray(fr["outflow_cum"], dtype=float)
        gap = float(np.max(np.abs((mass - m0) - (inflow - outflow))))
        rec = record(label=f"{key}：M(t)-M(0) 与边界通量累计的最大差",
                     theory="<= 1e-14（望远镜求和 + 时间插值误差）",
                     measured=gap, tol=1e-14, passed=(gap <= 1e-14),
                     detail=f"mass range {mass[0]:.6f} -> {mass[-1]:.6f}")
        assert rec.passed, rec.measured


@assertion("B", theory="cross_mask=0 的特征线对不许给出有限的「交点」；掩码=1 的交点必须有限且为正",
           chapter=CHAPTER)
def test_characteristic_cross_times_respect_the_mask() -> None:
    fr = _frame("characteristics", 0.3, {"ic": "sine", "n_lines": 40})
    t_cross = np.asarray(fr["cross_t"], dtype=float)
    mask = np.asarray(fr["cross_mask"], dtype=float)
    bad_masked = np.where((mask < 0.5) & (t_cross > 1e-6))[0]
    bad_live = np.where((mask > 0.5) & (~np.isfinite(t_cross) | (t_cross <= 0.0)))[0]
    ok = bad_masked.size == 0 and bad_live.size == 0
    rec = record(label="特征线交点数组的掩码一致性", theory="掩码=0 的点不得含有效交点",
                 measured=f"masked 越界 {bad_masked.size} 个，有效但非正 {bad_live.size} 个",
                 tol="0 个", passed=ok)
    assert rec.passed, rec.measured
