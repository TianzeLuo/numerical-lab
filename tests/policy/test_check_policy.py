"""策略检查器自身的测试（元测试）。

一个"永远返回 0 违规"的检查器和没有检查器是一回事。因此这里对每条规则都做**双向**测试：

* 正例：构造真正违规的代码 → 必须报；
* 反例：构造合法但"长得像违规"的代码（如 ``np.dot``、``reference/`` 里的 scipy） → 必须不报。

最后还有一条集成测试：**当前仓库必须 0 违规**（这就是 CI 门禁 G1 本身）。
"""

from __future__ import annotations

import json
import pathlib
import textwrap

import pytest

from tools.check_policy import (
    P6_EXEMPT_FILES,
    Violation,
    check,
    check_package_json,
    scan_source,
    scan_ts_source,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]


def rules_of(violations: list[Violation]) -> set[str]:
    return {v.rule for v in violations}


def scan(src: str, path: str) -> list[Violation]:
    return scan_source(textwrap.dedent(src), path=path)


# --------------------------------------------------------------------------- #
# P1 / P2：现成求解器只许出现在 numcore/reference/
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "src",
    [
        "import scipy.linalg",
        "from scipy import linalg",
        "from scipy.optimize import brentq",
        "import importlib\nimportlib.import_module('scipy.linalg')",
    ],
)
def test_p1_scipy_forbidden_outside_reference(src: str) -> None:
    assert "P1" in rules_of(scan(src, "numcore/linalg/lu.py"))
    assert "P1" in rules_of(scan(src, "tests/1.2/test_lu.py"))
    assert "P1" in rules_of(scan(src, "tools/foo.py"))
    assert "P1" not in rules_of(scan(src, "numcore/reference/dense.py"))


@pytest.mark.parametrize(
    "src",
    [
        "import numpy.linalg as la\nla.solve(A, b)",
        "from numpy.linalg import solve",
        "import numpy as np\nnp.linalg.solve(A, b)",
        "import numpy as np\nnp.linalg.norm(x)",
        "import numpy as np\nnp.fft.fft(x)",
        "import numpy.fft",
    ],
)
def test_p2_numpy_linalg_and_fft_are_global_red_lines(src: str) -> None:
    assert "P2" in rules_of(scan(src, "numcore/transform/fft.py"))
    assert "P2" in rules_of(scan(src, "tests/1.2/test_lu.py"))
    assert "P2" in rules_of(scan(src, "server/app.py"))
    assert "P2" not in rules_of(scan(src, "numcore/reference/dense.py"))


def test_p2_does_not_fire_on_own_modules() -> None:
    """自家模块名里带 linalg/fft 不是违规（避免把项目自己的名字误判）。"""
    src = """
    from numcore.linalg import lu
    from numcore.transform import fft
    x = lu.lu(A)
    """
    assert rules_of(scan(src, "numcore/pde/heat.py")) == set()


# --------------------------------------------------------------------------- #
# P3：numcore 不得使用 numpy 成品算法（但原语随便用）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "src",
    [
        "import numpy as np\nnp.polyfit(x, y, 3)",
        "import numpy as np\nnp.trapz(y, x)",
        "import numpy as np\nnp.gradient(u, dx)",
        "import numpy as np\nnp.interp(xq, x, y)",
        "import numpy as np\nnp.convolve(a, b)",
        "from numpy.polynomial import legendre",
    ],
)
def test_p3_ready_made_algorithms_forbidden_in_numcore(src: str) -> None:
    assert "P3" in rules_of(scan(src, "numcore/interp/poly.py"))


@pytest.mark.parametrize(
    "src",
    [
        "import numpy as np\nA @ B",
        "import numpy as np\nnp.dot(a, b)",
        "import numpy as np\nnp.einsum('ij,jk->ik', A, B)",
        "import numpy as np\nnp.diff(v)",
        "import numpy as np\nnp.cumsum(v)",
        "import numpy as np\nnp.outer(a, b)",
        "import numpy as np\nnp.clip(x, 0, 1)",
        "import numpy as np\nnp.argmax(v)",
    ],
)
def test_p3_primitives_are_allowed(src: str) -> None:
    assert "P3" not in rules_of(scan(src, "numcore/linalg/lu.py"))


def test_p3_only_applies_inside_numcore() -> None:
    """测试与工具里可以用 numpy 的现成算法做交叉验证（它们不是交付的自研实现）。"""
    src = "import numpy as np\nnp.polyfit(x, y, 2)\nnp.trapz(y, x)"
    assert "P3" not in rules_of(scan(src, "tests/2.1/test_poly.py"))
    assert "P3" not in rules_of(scan(src, "tools/raster.py"))


# --------------------------------------------------------------------------- #
# P4：随机数必须走 numcore.utils.rng
# --------------------------------------------------------------------------- #
def test_p4_random_is_centralised() -> None:
    src = "import numpy as np\nrng = np.random.default_rng(0)"
    assert "P4" in rules_of(scan(src, "numcore/inverse/uq/mc.py"))
    assert "P4" in rules_of(scan(src, "tests/5.4/test_mc.py"))
    assert "P4" not in rules_of(scan(src, "numcore/utils.py"))  # 唯一合法的封装处
    assert "P4" not in rules_of(scan(src, "numcore/reference/dense.py"))


# --------------------------------------------------------------------------- #
# P5：重库只在 reference 与 tests 里
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mod", ["sympy", "mpmath", "matplotlib", "pandas", "sklearn", "numba"])
def test_p5_heavy_libs_forbidden_in_numcore(mod: str) -> None:
    assert "P5" in rules_of(scan(f"import {mod}", "numcore/interp/poly.py"))
    assert "P5" not in rules_of(scan(f"import {mod}", "numcore/reference/exact.py"))
    assert "P5" not in rules_of(scan(f"import {mod}", "tests/test_reference_layer.py"))


# --------------------------------------------------------------------------- #
# P6：numcore 的纯函数约束
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "src,exc",
    [
        ("print('debug')", "print"),
        ("open('f')", "open"),
        ("import os", "os"),
        ("from pathlib import Path", "pathlib"),
    ],
)
def test_p6_io_and_print_forbidden(src: str, exc: str) -> None:
    got = scan(src, "numcore/pde/heat.py")
    assert "P6" in rules_of(got), exc


def test_p6_global_statement_forbidden() -> None:
    src = """
    _CACHE = {}

    def f(x):
        global _CACHE
        return x
    """
    assert "P6" in rules_of(scan(src, "numcore/linalg/lu.py"))


def test_p6_exempts_test_infrastructure_files() -> None:
    """pytest 插件与报告落盘是测试基础设施，其 IO 被显式豁免（见 POLICY §1 例外登记）。"""
    src = "import os\nfrom pathlib import Path\nprint('x')"
    for f in P6_EXEMPT_FILES:
        assert "P6" not in rules_of(scan(src, f)), f
    assert "P6" in rules_of(scan(src, "numcore/verify/order.py"))  # 同包其他文件不豁免


# --------------------------------------------------------------------------- #
# P10：参照物不得被生产代码引用
# --------------------------------------------------------------------------- #
def test_p10_reference_layer_is_test_only() -> None:
    src = "from numcore.reference import dense as ref"
    assert "P10" in rules_of(scan(src, "numcore/linalg/lu.py"))
    assert "P10" in rules_of(scan(src, "server/app.py"))
    assert "P10" not in rules_of(scan(src, "tests/1.2/test_lu.py"))
    assert "P10" not in rules_of(scan(src, "tools/parity.py"))
    assert "P10" not in rules_of(scan(src, "numcore/reference/dense.py"))


# --------------------------------------------------------------------------- #
# P7 / P8 / P12：前端规则
# --------------------------------------------------------------------------- #
def test_p7_frontend_has_no_runtime_dependencies() -> None:
    assert rules_of(scan_ts_source('import * as THREE from "three";', path="viz/shared/src/a.ts")) == {"P7"}
    assert rules_of(scan_ts_source('import { x } from "./b.js";', path="viz/shared/src/a.ts")) == set()
    assert rules_of(scan_ts_source('import { test } from "node:test";', path="viz/tests/a.test.ts")) == set()
    assert rules_of(scan_ts_source('import fs from "node:fs";', path="viz/shared/src/a.ts")) == {"P7"}


def test_p8_chapter_frontend_must_declare_preview_usage() -> None:
    bad = 'import { lu } from "../../shared/src/numeric/lu.js";\n'
    assert "P8" in rules_of(scan_ts_source(bad, path="viz/chapters/1.2/a.model.ts"))
    ok = '// preview-approx: 仅用于拖动时的近似预览，最终数据来自后端\nimport { lu } from "../../shared/src/numeric/lu.js";\n'
    assert "P8" not in rules_of(scan_ts_source(ok, path="viz/chapters/1.2/a.model.ts"))
    # 不碰 numeric/ 的章节文件不受此规则影响
    assert rules_of(scan_ts_source('import { decodeNlf } from "../../shared/src/nlf.js";', path="viz/chapters/1.2/a.model.ts")) == set()


def test_p12_generated_files_must_be_marked() -> None:
    assert "P12" in rules_of(scan_ts_source("export const A = 1;\n", path="viz/chapters/1.1/manifest.gen.ts"))
    marked = "// ⚠️ 本文件由 tools/gen_manifests.py 生成，请勿手改。\nexport const A = 1;\n"
    assert "P12" not in rules_of(scan_ts_source(marked, path="viz/chapters/1.1/manifest.gen.ts"))


# --------------------------------------------------------------------------- #
# package.json
# --------------------------------------------------------------------------- #
def test_package_json_dependencies(tmp_path: pathlib.Path) -> None:
    (tmp_path / "package.json").write_text(json.dumps({"dependencies": {"three": "^1"}}), encoding="utf-8")
    assert "P7" in rules_of(check_package_json(tmp_path))
    (tmp_path / "package.json").write_text(
        json.dumps({"devDependencies": {"typescript": "^5", "vitest": "^2"}}), encoding="utf-8"
    )
    assert "P7" in rules_of(check_package_json(tmp_path))
    (tmp_path / "package.json").write_text(
        json.dumps({"devDependencies": {"typescript": "^5", "@types/node": "^22"}}), encoding="utf-8"
    )
    assert rules_of(check_package_json(tmp_path)) == set()


# --------------------------------------------------------------------------- #
# 行内豁免必须登记
# --------------------------------------------------------------------------- #
def _make_repo(tmp_path: pathlib.Path, body: str, allowlist: str | None) -> pathlib.Path:
    (tmp_path / "numcore" / "linalg").mkdir(parents=True)
    (tmp_path / "numcore" / "linalg" / "lu.py").write_text(textwrap.dedent(body), encoding="utf-8")
    (tmp_path / "tests" / "policy").mkdir(parents=True)
    if allowlist is not None:
        (tmp_path / "tests" / "policy" / "allowlist.txt").write_text(allowlist, encoding="utf-8")
    return tmp_path


def test_unregistered_inline_allowance_is_a_violation(tmp_path: pathlib.Path) -> None:
    body = """
    import scipy.linalg  # policy: allow P1 临时用来验证一个边界情形
    """
    repo = _make_repo(tmp_path, body, allowlist=None)
    violations, _ = check(repo)
    assert {v.rule for v in violations} == {"ALLOWLIST"}, violations
    assert "未登记的豁免" in violations[0].message


def test_registered_inline_allowance_suppresses_rule(tmp_path: pathlib.Path) -> None:
    body = """
    import scipy.linalg  # policy: allow P1 需要读取一个 scipy 特有的格式
    """
    repo = _make_repo(tmp_path, body, allowlist="numcore/linalg/lu.py:P1\n")
    violations, _ = check(repo)
    assert violations == [], violations


def test_registration_without_use_is_harmless(tmp_path: pathlib.Path) -> None:
    """登记了但没写行内注释：不报违规（登记表允许"富余"）。"""
    body = "x = 1\n"
    repo = _make_repo(tmp_path, body, allowlist="numcore/linalg/lu.py:P1\n")
    violations, _ = check(repo)
    assert violations == []


# --------------------------------------------------------------------------- #
# 语法错误也要被抓住
# --------------------------------------------------------------------------- #
def test_syntax_error_is_reported() -> None:
    got = scan("def f(:\n    pass", "numcore/x.py")
    assert {v.rule for v in got} == {"SYNTAX"}


# --------------------------------------------------------------------------- #
# 集成：当前仓库必须 0 违规（= CI 门禁 G1）
# --------------------------------------------------------------------------- #
def test_repository_is_policy_clean() -> None:
    violations, notes = check(ROOT)
    assert violations == [], "\n".join(str(v) for v in violations)
    assert any("P6 文件级豁免" in n for n in notes)
