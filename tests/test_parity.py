"""``tests/test_parity.py`` —— G7 双实现对拍门禁的元测试（PROTOCOLS §7）。

一条门禁只有在**自身可测**时才算数。这里逐条钉住的失败模式：

1. **golden 过期**：`--update-golden` 必须逐字节可重生成（无时间戳、键排序、LF），
   否则"基准数据"会随时间悄悄漂移，而门禁照样绿灯；
2. **索引漂移**：`cases.json` ↔ `tests/parity/golden/*.json` 必须一一对应（不多不少），
   校验和必须对得上磁盘字节；
3. **容差配置错**：覆盖必须与 case 集合双向一致，且每一条都得写 `reason`；
4. **静默通过**：TS 报告缺失必须报错并给出构建命令；NaN/Inf 必须判失败（不能因为
   "两边都是 NaN"就放行）；恰好等于容差必须**通过**（判定用 `<=`）；
5. **报错不达标**：失败输出必须含 PROTOCOLS §7 要求的六项
   （`case` / 索引 / `python` 值 / `ts` 值 / 相对差 / 容差）与"最大超差 Top 5"。
"""

from __future__ import annotations

import copy
import json
import math
import pathlib
import shutil
import subprocess
import sys

import pytest

import tools.parity as parity

pytestmark = pytest.mark.parity

REPO = pathlib.Path(__file__).resolve().parents[1]

#: 报告格式里最小可用的合成 case（只测比较器，不碰真实算法）。
SYNTHETIC_SPEC = {"name": "synthetic_case", "algo": "heat", "outputs": ["u"]}


# --------------------------------------------------------------------------- #
# 夹具与工具
# --------------------------------------------------------------------------- #
@pytest.fixture()
def tmp_repo(tmp_path: pathlib.Path) -> pathlib.Path:
    """临时仓库根：把 ``tests/parity`` 原样复制过去，``dist/`` 交给被测代码自己建。"""
    dst = tmp_path / "repo"
    (dst / "tests").mkdir(parents=True)
    shutil.copytree(REPO / "tests" / "parity", dst / "tests" / "parity")
    return dst


def _python_outputs() -> dict[str, dict]:
    """全部 case 的 Python 侧输出（深拷贝，供伪造 TS 报告用）。"""
    return {spec["name"]: copy.deepcopy(parity.python_outputs(spec)) for spec in parity.CASE_SPECS}


def _write_ts_report(
    root: pathlib.Path,
    outputs: dict[str, dict] | None = None,
    *,
    drop: tuple[str, ...] = (),
    errors: dict[str, str] | None = None,
) -> pathlib.Path:
    """按 runner 的格式写一份 ``parity_ts.json``（缺省是与 Python 完全一致的"诚实"报告）。"""
    data = _python_outputs() if outputs is None else outputs
    cases: dict[str, dict] = {}
    for name, value in data.items():
        if name in drop:
            continue
        cases[name] = {"algo": parity.case_by_name(name)["algo"], "outputs": value}
    for name, message in (errors or {}).items():
        cases[name] = {"algo": parity.case_by_name(name)["algo"], "error": message}
    path = root / parity.TS_REPORT_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"generated_at": "2026-10-18T00:00:00Z", "cases": cases}), encoding="utf-8")
    return path


def _golden_text(name: str) -> str:
    return (REPO / parity.GOLDEN_REL / f"{name}.json").read_text(encoding="utf-8")


def _cases_index() -> dict:
    return json.loads((REPO / parity.CASES_REL).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# 1. case 索引与「不多不少」
# --------------------------------------------------------------------------- #
def test_protocol_required_cases_are_present() -> None:
    names = {spec["name"] for spec in parity.CASE_SPECS}
    assert {
        "heat_explicit_step",
        "heat_ftcs_run",
        "ftcs_amplification_modes",
        "oscillator_sample",
        "order_fit_powerlaw",
    } <= names
    assert len(names) >= 5


def test_case_names_are_unique_and_algos_supported() -> None:
    names = [spec["name"] for spec in parity.CASE_SPECS]
    assert len(names) == len(set(names))
    for spec in parity.CASE_SPECS:
        assert spec["algo"] in parity.ALGOS, spec["name"]
        assert spec["outputs"], f"{spec['name']} 没有声明任何输出键"


def test_cases_index_matches_golden_files_one_to_one() -> None:
    """``cases.json`` 与 golden 目录一一对应：不少一个、也不多一个（nlf_* 属于 §1，显式排除）。"""
    index = _cases_index()
    declared = {pathlib.Path(case["golden"]).name for case in index["cases"]}
    on_disk = {
        path.name
        for path in (REPO / parity.GOLDEN_REL).glob("*.json")
        if not path.name.startswith(parity.FOREIGN_GOLDEN_PREFIXES)
    }
    assert declared == on_disk
    assert len(declared) == len(parity.CASE_SPECS)


def test_cases_index_checksums_match_disk_bytes() -> None:
    """`sha256` 是"golden 没有被手改过"的锚点。"""
    import hashlib

    for case in _cases_index()["cases"]:
        blob = (REPO / case["golden"]).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == case["sha256"], case["name"]


def test_cases_index_declares_same_outputs_as_golden() -> None:
    for case in _cases_index()["cases"]:
        golden = json.loads((REPO / case["golden"]).read_text(encoding="utf-8"))
        assert case["name"] == golden["case"]
        assert case["algo"] == golden["algo"]
        # JSON 对象键在落盘时被排序，两侧只比集合（顺序不承载语义）
        assert set(case["outputs"]) == set(golden["outputs"].keys())


def test_every_algo_has_a_ts_runner_branch() -> None:
    """runner 的算法分支表与 Python 侧必须同名（否则 case 会被静默跳过）。"""
    source = (REPO / "viz" / "tests" / "parity" / "run_parity.ts").read_text(encoding="utf-8")
    for algo in parity.ALGOS:
        assert f"{algo}:" in source, f"runner 里缺少 algo={algo} 的分支"


# --------------------------------------------------------------------------- #
# 2. golden 的可重生成性与确定性
# --------------------------------------------------------------------------- #
def test_golden_regeneration_is_byte_identical(tmp_path: pathlib.Path) -> None:
    """在临时根重跑 `--update-golden`，产物必须与仓库里的**逐字节**相同。"""
    root = tmp_path / "repo"
    proc = subprocess.run(
        [sys.executable, str(REPO / "tools" / "parity.py"), "--update-golden", "--root", str(root), "--quiet"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    for spec in parity.CASE_SPECS:
        rel = parity.GOLDEN_REL / f"{spec['name']}.json"
        assert (root / rel).read_bytes() == (REPO / rel).read_bytes(), f"{rel} 不可重生成"
    assert (root / parity.CASES_REL).read_bytes() == (REPO / parity.CASES_REL).read_bytes()


def test_golden_regeneration_is_idempotent(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "repo"
    cmd = [sys.executable, str(REPO / "tools" / "parity.py"), "--update-golden", "--root", str(root), "--quiet"]
    subprocess.run(cmd, cwd=str(REPO), check=True)
    first = {p.relative_to(root): p.read_bytes() for p in sorted(root.rglob("*.json"))}
    subprocess.run(cmd, cwd=str(REPO), check=True)
    second = {p.relative_to(root): p.read_bytes() for p in sorted(root.rglob("*.json"))}
    assert first == second


def test_golden_files_are_lf_without_cr() -> None:
    for spec in parity.CASE_SPECS:
        blob = (REPO / parity.GOLDEN_REL / f"{spec['name']}.json").read_bytes()
        assert b"\r" not in blob, spec["name"]
        assert blob.endswith(b"\n") and not blob.endswith(b"\n\n"), spec["name"]
    cases_blob = (REPO / parity.CASES_REL).read_bytes()
    assert b"\r" not in cases_blob and cases_blob.endswith(b"\n")


def test_golden_files_carry_no_wall_clock() -> None:
    """golden 是对拍契约，掺进时间戳就永远不可重生成。"""
    import re

    stamp = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")
    for spec in parity.CASE_SPECS:
        text = _golden_text(spec["name"])
        assert "generated_at" not in text, spec["name"]
        assert not stamp.search(text), spec["name"]


def test_golden_text_round_trips_without_precision_loss() -> None:
    """读进来再按同一规则写出去必须逐字节相同 —— 证明 double 没有被打印精度截断。"""
    for spec in parity.CASE_SPECS:
        text = _golden_text(spec["name"])
        assert parity.dumps_canonical(json.loads(text)) == text, spec["name"]


def test_python_outputs_reproduce_disk_golden_exactly() -> None:
    """`numcore` 现在的输出必须与磁盘 golden 逐位相同（golden 未过期的最强形式）。"""
    for spec in parity.CASE_SPECS:
        golden = json.loads(_golden_text(spec["name"]))
        assert parity.python_outputs(spec) == golden["outputs"], spec["name"]


def test_build_artifacts_is_pure() -> None:
    """同一进程内连算两次结果必须一致（没有隐藏的随机/时间依赖）。"""
    _, first, cases_first, _ = parity.build_artifacts()
    _, second, cases_second, _ = parity.build_artifacts()
    assert first == second
    assert cases_first == cases_second


# --------------------------------------------------------------------------- #
# 3. golden 的数值契约（与 numcore、与物理）
# --------------------------------------------------------------------------- #
def test_heat_cases_use_the_requested_step_counts() -> None:
    """`heat_ftcs` 的 `ceil(t_end/Δt)` 必须正好落在目标步数上，否则基准数据就是错的。"""
    single = parity.case_by_name("heat_explicit_step")
    run = parity.case_by_name("heat_ftcs_run")
    _, outputs_single, meta_single = parity._build_heat(single["params"])
    _, _, meta_run = parity._build_heat(run["params"])
    assert meta_single["n_steps"] == 1
    assert meta_run["n_steps"] == 40
    assert len(outputs_single["u"]) == single["params"]["nx"]


def test_heat_golden_keeps_dirichlet_ends_at_zero() -> None:
    for name in ("heat_explicit_step", "heat_ftcs_run"):
        golden = json.loads(_golden_text(name))
        u = golden["outputs"]["u"]
        assert u[0] == 0.0 and u[-1] == 0.0
        assert max(u[1:-1]) < max(golden["inputs"]["u0"][1:-1])  # 显式格式在稳定区应当衰减


def test_heat_initial_profile_matches_golden_input() -> None:
    golden = json.loads(_golden_text("heat_explicit_step"))
    assert golden["inputs"]["u0"] == parity._build_heat(parity.case_by_name("heat_explicit_step")["params"])[0]["u0"]
    # 注意：初值剖面本身只保证 sin(π·1) ≈ 1.2e-16，两端真正被钉成 0 的是**步进**
    assert golden["inputs"]["u0"][0] == 0.0
    assert abs(golden["inputs"]["u0"][-1]) < 1e-15


def test_amplification_golden_respects_stability_interval() -> None:
    """r = 0.3 < 1/2，全部可分辨模态必须 |g| < 1（金标准本身要落在物理上该在的区间里）。"""
    golden = json.loads(_golden_text("ftcs_amplification_modes"))
    assert all(abs(g) < 1.0 for g in golden["outputs"]["g"])
    assert len(golden["outputs"]["g"]) == len(golden["inputs"]["modes"]) == 8


def test_oscillator_golden_covers_three_regimes() -> None:
    golden = json.loads(_golden_text("oscillator_sample"))
    for tag in ("under", "crit", "over"):
        assert f"x_{tag}" in golden["outputs"] and f"v_{tag}" in golden["outputs"]
        assert len(golden["outputs"][f"x_{tag}"]) == golden["meta"]["n_samples"] == 200
    zetas = [branch["zeta"] for branch in golden["inputs"]["branches"]]
    assert zetas == [0.05, 1.0, 1.5]
    x_under = golden["outputs"]["x_under"]
    x_over = golden["outputs"]["x_over"]
    assert any(abs(a - b) > 1e-3 for a, b in zip(x_under, x_over)), "三条分支不该算出同一条曲线"


def test_order_fit_golden_recovers_synthetic_power_laws() -> None:
    """人造幂律的斜率必须被拟回理论阶（±0.1），否则"基准"本身就是错的。"""
    golden = json.loads(_golden_text("order_fit_powerlaw"))
    out = golden["outputs"]
    assert out["slope_pure_p2"] == pytest.approx(2.0, abs=0.1)
    assert out["slope_p3_head7"] == pytest.approx(3.0, abs=0.1)
    assert out["slope_p05_skip1_tail9"] == pytest.approx(0.5, abs=0.1)
    assert out["nUsed_p3_head7"] == 7 and out["nDropped_p3_head7"] == 5
    # 舍入地板那一组：只取粗端 6 点，否则斜率会被地板拖低
    assert out["slope_rounding_floor_head6"] == pytest.approx(2.0, abs=0.1)
    assert out["nUsed_rounding_floor_head6"] == 6


# --------------------------------------------------------------------------- #
# 4. 容差文件
# --------------------------------------------------------------------------- #
def test_default_tolerance_matches_protocol() -> None:
    default, _ = parity.load_tolerances(REPO / parity.TOL_REL)
    assert default.rtol == 1e-12 and default.atol == 1e-14
    assert default.source == "default"


def test_tolerance_overrides_and_cases_match_exactly() -> None:
    _, overrides = parity.load_tolerances(REPO / parity.TOL_REL)
    keys = {spec["tolerance"] for spec in parity.CASE_SPECS}
    assert set(overrides) == keys, "容差覆盖与 case 集合必须双向一致（不多不少）"


def test_every_tolerance_override_has_reason() -> None:
    default, overrides = parity.load_tolerances(REPO / parity.TOL_REL)
    assert default.reason.strip()
    for name, tol in overrides.items():
        assert tol.reason.strip(), f"{name} 的容差没有 reason"
        assert tol.rtol > 0.0 or tol.atol > 0.0


def test_tolerance_file_rejects_override_without_reason(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "tolerances.yaml"
    path.write_text(
        "default: {rtol: 1e-12, atol: 1e-14, reason: '默认'}\n"
        "cases:\n"
        "  heat_explicit_step: {rtol: 1e-15, atol: 1e-16}\n",
        encoding="utf-8",
    )
    with pytest.raises(parity.ParityError, match="reason"):
        parity.load_tolerances(path)


def test_tolerance_file_rejects_unknown_keys_and_zero_tolerance(tmp_path: pathlib.Path) -> None:
    bad_key = tmp_path / "bad_key.yaml"
    bad_key.write_text("default: {rtol: 1e-12, atol: 1e-14, reason: 'x', extra: 1}\n", encoding="utf-8")
    with pytest.raises(parity.ParityError, match="未知键"):
        parity.load_tolerances(bad_key)
    zero = tmp_path / "zero.yaml"
    zero.write_text("default: {rtol: 0, atol: 0, reason: 'x'}\n", encoding="utf-8")
    with pytest.raises(parity.ParityError, match="rtol=atol=0"):
        parity.load_tolerances(zero)


# --------------------------------------------------------------------------- #
# 5. 判定口径（边界 / 非有限值 / 缺失）
# --------------------------------------------------------------------------- #
def _compare(py: dict, ts: dict, tol: parity.Tolerance, spec: dict | None = None):
    return parity.compare_case(spec or SYNTHETIC_SPEC, py, ts, tol)


def test_exact_tolerance_boundary_passes() -> None:
    """恰好等于容差 → 通过（判定是 `<=`，不是 `<`）。"""
    tol = parity.Tolerance(rtol=0.0, atol=1e-14)
    mismatches, stats = _compare({"u": [0.0]}, {"u": [1e-14]}, tol)
    assert mismatches == []
    assert stats["max_ratio"] == pytest.approx(1.0)


def test_one_ulp_beyond_tolerance_fails() -> None:
    tol = parity.Tolerance(rtol=0.0, atol=1e-14)
    just_over = math.nextafter(1e-14, math.inf)
    mismatches, _ = _compare({"u": [0.0]}, {"u": [just_over]}, tol)
    assert len(mismatches) == 1
    assert mismatches[0].ratio > 1.0
    assert mismatches[0].key == "u" and mismatches[0].index == "-"  # 标量输出没有索引维度


def test_rtol_scales_with_python_value() -> None:
    tol = parity.Tolerance(rtol=1e-12, atol=0.0)
    assert _compare({"u": [1e6]}, {"u": [1e6 + 5e-7]}, tol)[0] == []
    assert len(_compare({"u": [1e6]}, {"u": [1e6 + 2e-6]}, tol)[0]) == 1


@pytest.mark.parametrize("py_value,ts_value", [(math.nan, math.nan), (math.nan, 1.0), (math.inf, math.inf)])
def test_non_finite_values_always_fail(py_value: float, ts_value: float) -> None:
    """NaN/Inf 不参与数值比较，必须直接判失败（绝不允许"两边都是 NaN"式静默通过）。"""
    tol = parity.Tolerance(rtol=1e-12, atol=1e-14)
    mismatches, _ = _compare({"u": [py_value]}, {"u": [ts_value]}, tol)
    assert len(mismatches) == 1
    assert "非有限值" in mismatches[0].note


def test_ts_side_nan_on_finite_python_fails() -> None:
    tol = parity.Tolerance(rtol=1e-12, atol=1e-14)
    mismatches, _ = _compare({"u": [0.5]}, {"u": [math.nan]}, tol)
    assert len(mismatches) == 1 and "ts" in mismatches[0].note


def test_length_mismatch_is_reported() -> None:
    tol = parity.Tolerance(rtol=1e-12, atol=1e-14)
    mismatches, _ = _compare({"u": [1.0, 2.0]}, {"u": [1.0, 2.0, 3.0]}, tol)
    assert len(mismatches) == 1 and "长度不一致" in mismatches[0].note


def test_missing_output_key_is_reported() -> None:
    tol = parity.Tolerance(rtol=1e-12, atol=1e-14)
    mismatches, _ = _compare({"u": [1.0]}, {}, tol)
    assert len(mismatches) == 1 and "缺少该输出键" in mismatches[0].note


def test_missing_case_in_ts_report_is_reported() -> None:
    tol = parity.Tolerance(rtol=1e-12, atol=1e-14)
    mismatches, stats = _compare({"u": [1.0]}, None, tol)
    assert len(mismatches) == 1 and "没有这个 case" in mismatches[0].note
    assert stats["n_mismatch"] == 1


def test_non_numeric_ts_output_is_reported() -> None:
    tol = parity.Tolerance(rtol=1e-12, atol=1e-14)
    mismatches, _ = _compare({"u": [1.0]}, {"u": ["1.0"]}, tol)
    assert len(mismatches) == 1 and "不是数值" in mismatches[0].note


def test_identical_outputs_pass_cleanly() -> None:
    """恒等对照：comparator 不能有"自己和自己比也超差"的毛病。"""
    tol = parity.Tolerance(rtol=1e-12, atol=1e-14)
    for spec in parity.CASE_SPECS:
        outputs = parity.python_outputs(spec)
        mismatches, stats = parity.compare_case(spec, outputs, outputs, tol)
        assert mismatches == []
        assert stats["n_points"] == sum(
            len(v) if isinstance(v, list) else 1 for v in outputs.values()
        )


# --------------------------------------------------------------------------- #
# 6. TS 报告缺失 / 超差报错格式 / Top 5
# --------------------------------------------------------------------------- #
def test_missing_ts_report_raises_with_build_hint(tmp_repo: pathlib.Path) -> None:
    with pytest.raises(parity.ParityError) as excinfo:
        parity.load_ts_report(tmp_repo, auto_run=False)
    message = str(excinfo.value)
    assert parity.TS_REPORT_REL.as_posix() in message
    assert parity.BUILD_HINT in message


def test_default_mode_fails_loudly_when_ts_report_missing(tmp_repo: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
    code = parity.main(["--root", str(tmp_repo), "--no-auto-run"])
    captured = capsys.readouterr()
    assert code == 1
    assert parity.BUILD_HINT in (captured.err + captured.out)


def test_failure_output_has_all_required_fields(tmp_repo: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
    """伪造一份 TS 报告：必须在 stdout 里看到六项（case/索引/python/ts/相对差/容差）。"""
    outputs = _python_outputs()
    outputs["heat_explicit_step"]["u"][3] *= 1.0 + 1e-6
    _write_ts_report(tmp_repo, outputs)
    code = parity.main(["--root", str(tmp_repo), "--no-auto-run"])
    out = capsys.readouterr().out
    assert code == 1
    for token in ("case=heat_explicit_step", "index=u[3]", "python=", "ts=", "rel_diff=", "tol=rtol="):
        assert token in out, f"失败输出缺少必需字段：{token}"
    assert "最大超差 Top 5" in out


def test_worst_five_are_sorted_and_are_the_largest(tmp_repo: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
    """7 处超差 → Top 5 只留最大的 5 个，且按超差倍数降序。"""
    outputs = _python_outputs()
    factors = [1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6]
    for index, factor in zip(range(1, 8), factors):
        outputs["heat_explicit_step"]["u"][index] *= 1.0 + factor
    _write_ts_report(tmp_repo, outputs)
    code = parity.main(["--root", str(tmp_repo), "--no-auto-run", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 1
    assert report["n_mismatch"] == 7
    assert len(report["worst"]) == 5
    ratios = [item["ratio"] for item in report["worst"]]
    assert ratios == sorted(ratios, reverse=True)
    indexes = [item["index"] for item in report["worst"]]
    assert indexes == ["7", "6", "5", "4", "3"], indexes
    assert (tmp_repo / parity.REPORT_REL).exists()


def test_ts_runner_error_is_surfaced_as_failure(tmp_repo: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
    outputs = _python_outputs()
    outputs.pop("oscillator_sample")
    _write_ts_report(tmp_repo, outputs, errors={"oscillator_sample": "有效点不足：给定 12 点"})
    code = parity.main(["--root", str(tmp_repo), "--no-auto-run", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 1
    case = next(item for item in report["cases"] if item["name"] == "oscillator_sample")
    assert case["ts_error"] == "有效点不足：给定 12 点"
    assert any("TS runner 报错" in item["note"] for item in case["mismatches"])


def test_missing_case_in_ts_report_is_surfaced(tmp_repo: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
    outputs = _python_outputs()
    outputs.pop("heat_ftcs_run")
    _write_ts_report(tmp_repo, outputs)
    code = parity.main(["--root", str(tmp_repo), "--no-auto-run", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 1
    assert report["missing_ts_cases"] == ["heat_ftcs_run"]


def test_json_mode_is_pure_json_and_exit_zero_when_honest(tmp_repo: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
    _write_ts_report(tmp_repo)
    code = parity.main(["--root", str(tmp_repo), "--no-auto-run", "--json"])
    out = capsys.readouterr().out
    report = json.loads(out)  # 必须是纯 JSON，不能掺人读文字
    assert code == 0 and report["ok"] is True
    assert report["n_mismatch"] == 0 and report["n_points"] > 1000
    assert (tmp_repo / parity.REPORT_REL).exists()


# --------------------------------------------------------------------------- #
# 7. --check-golden 的过期检测
# --------------------------------------------------------------------------- #
def test_check_golden_passes_on_repo() -> None:
    assert parity.check_artifacts(REPO) == []
    assert parity.main(["--check-golden", "--quiet"]) == 0


def test_check_golden_detects_tampered_golden(tmp_repo: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
    path = tmp_repo / parity.GOLDEN_REL / "heat_explicit_step.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["outputs"]["u"][3] *= 1.0000001
    path.write_text(parity.dumps_canonical(doc), encoding="utf-8")
    stale = parity.check_artifacts(tmp_repo)
    assert any("heat_explicit_step.json" in item for item in stale)
    assert parity.main(["--check-golden", "--root", str(tmp_repo)]) == 1
    assert "过期" in capsys.readouterr().out


def test_check_golden_detects_missing_and_extra_files(tmp_repo: pathlib.Path) -> None:
    (tmp_repo / parity.GOLDEN_REL / "heat_ftcs_run.json").unlink()
    stale = parity.check_artifacts(tmp_repo)
    assert any("缺失" in item and "heat_ftcs_run.json" in item for item in stale)

    (tmp_repo / parity.GOLDEN_REL / "orphan_case.json").write_text("{}\n", encoding="utf-8")
    stale = parity.check_artifacts(tmp_repo)
    assert any("多余" in item and "orphan_case.json" in item for item in stale)
    # nlf_*.json 属于 PROTOCOLS §1，不该被本工具认领
    assert not any("nlf_" in item for item in stale)


def test_cases_index_checksum_catches_stale_golden(tmp_repo: pathlib.Path) -> None:
    """就算在脏工作区里跑（golden 被改），cases.json 的校验和也必须对不上。"""
    path = tmp_repo / parity.CASES_REL
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["cases"][0]["sha256"] = "0" * 64
    path.write_text(parity.dumps_canonical(doc), encoding="utf-8")
    stale = parity.check_artifacts(tmp_repo)
    assert any(parity.CASES_REL.as_posix() in item for item in stale)
