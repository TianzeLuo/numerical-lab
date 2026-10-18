"""pytest 全局配置：把仓库根加入 ``sys.path``，并挂载 ``numcore.verify`` 报告插件。"""

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from numcore.verify import plugin as _numcore_plugin  # noqa: E402


def pytest_configure(config):  # noqa: ANN001, ANN201
    """会话开始：清空断言登记表。"""
    _numcore_plugin.pytest_configure(config)


def pytest_runtest_setup(item):  # noqa: ANN001, ANN201
    """每个测试开始前记录 nodeid，使断言可归属到章节。"""
    _numcore_plugin.pytest_runtest_setup(item)


def pytest_sessionfinish(session, exitstatus):  # noqa: ANN001, ANN201
    """会话结束：落盘 ``dist/reports/assertions.json`` 并打印摘要。"""
    _numcore_plugin.pytest_sessionfinish(session, exitstatus)
