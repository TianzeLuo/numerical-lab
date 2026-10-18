"""确定性随机源。

``numcore`` 内禁止直接使用 ``numpy.random``（POLICY P4）：所有随机数必须经此模块取得，
以便 (a) 种子集中管理、(b) 断言可复现、(c) 静态检查能一刀切掉"随手乱撒"的随机性。

数学约定：返回的生成器使用 PCG64；同一 ``seed`` 在同一 numpy 版本下逐位可复现。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["rng", "seed_sequence", "spawn_seeds"]


def rng(seed: int | Sequence[int] | None = None) -> np.random.Generator:
    """返回确定性随机数生成器 ``Generator(PCG64(seed))``。

    参数
    ----
    seed
        整数种子；``None`` 表示使用固定默认种子 ``20261018``（本项目"开天辟地"日），
        仍然确定——这是刻意的：不允许出现不可复现的随机实验。
    """
    if seed is None:
        seed = 20261018
    if isinstance(seed, (list, tuple)):
        return np.random.Generator(np.random.PCG64(np.asarray(seed, dtype=np.uint64)))
    return np.random.Generator(np.random.PCG64(int(seed)))


def seed_sequence(seed: int, n: int) -> np.ndarray:
    """由父种子派生 ``n`` 个互不相同的子种子（用于并行蒙特卡洛）。"""
    ss = np.random.SeedSequence(int(seed))
    return np.array([int(c.generate_state(1, dtype=np.uint32)[0]) for c in ss.spawn(int(n))], dtype=np.uint64)


def spawn_seeds(seed: int, n: int) -> list[np.random.Generator]:
    """由父种子派生 ``n`` 个独立生成器。"""
    return [rng(int(s)) for s in seed_sequence(seed, n)]
