r"""6.1 章：Ghia--Ghia--Shin (1982) 基准表的完整性与边界。

这一文件只钉"靶子"本身，不跑求解器：

* 三列 Re=100/400/1000 的 u(y)/v(x) 表各有 17 个点，端点严格是 0/1；
* 竖直线 x=0.5 的 u、水平线 y=0.5 的 v，方向不能写反；
* 转载分歧 ``BENCHMARK_TRANSCRIPTION_SPREAD=1e-2`` 是任何"对拍容差"的下界；
* Re=1000 除了主涡还有左下/右下两个二次涡——只报主涡会漏掉一半结构。
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from numcore.apps import ghia  # noqa: E402
from numcore.verify import assertion, record  # noqa: E402


@assertion("A", theory="Ghia 三个 Re 的 u/v 表各 17 点，端点 u(0)=v(0)=v(1)=0、u(1)=1，采样单调", chapter="6.1")
def test_ghia_tables_complete_and_endpoints_exact() -> None:
    detail = []
    ok = True
    for re in ghia.REYNOLDS_SUPPORTED:
        y, u = ghia.centerline_u(re)
        x, v = ghia.centerline_v(re)
        ok = ok and y.size == 17 and u.size == 17 and x.size == 17 and v.size == 17
        ok = ok and np.all(np.diff(y) > 0) and np.all(np.diff(x) > 0)
        ok = ok and u[0] == 0.0 and u[-1] == 1.0 and v[0] == 0.0 and v[-1] == 0.0
        detail.append(f"Re={re}: u∈[{u.min():.5f},{u.max():.5f}]，v∈[{v.min():.5f},{v.max():.5f}]")
    rec = record(
        label="Ghia 三列的表长与端点",
        theory="每列 17 点；u(0)=0、u(1)=1、v(0)=v(1)=0；采样严格递增",
        measured="；".join(detail),
        tol="表长 17、端点严格相等、diff>0",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured


@assertion("A", theory="转载分歧 1e-2 是容差下界；Re=100/400/1000 的主涡位置随 Re 单调向中心移动", chapter="6.1")
def test_ghia_vortex_centers_and_transcription_floor() -> None:
    c100 = ghia.vortex_centers(100)["primary"]
    c400 = ghia.vortex_centers(400)["primary"]
    c1000 = ghia.vortex_centers(1000)
    primary = c1000["primary"]
    has_secondaries = "bottom_left" in c1000 and "bottom_right" in c1000
    ok = (
        c100[0] > c400[0] > primary[0]
        and c100[1] > c400[1] > primary[1]
        and has_secondaries
        and ghia.BENCHMARK_TRANSCRIPTION_SPREAD == pytest.approx(1.0e-2)
    )
    rec = record(
        label="主涡位置随 Re 的变化与转载分歧地板",
        theory="x、y 随 Re 单调减小；Re=1000 存在两个底角二次涡；分歧地板 1e-2",
        measured=(
            f"Re=100 主涡 {c100}，Re=400 {c400}，Re=1000 {primary}；"
            f"二次涡 {sorted(k for k in c1000 if k != 'primary')}；floor={ghia.BENCHMARK_TRANSCRIPTION_SPREAD:g}"
        ),
        tol="严格单调、二次涡存在、floor=1e-2",
        passed=bool(ok),
    )
    assert rec.passed, rec.measured
