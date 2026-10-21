#!/usr/bin/env bash
# 十道门禁一把跑完（G1–G10）。任何一道红 → 退出码 1，并把每道门的结果写进 dist/reports/ci.json。
#
# 设计要点：
#  * **不早停**：一次跑完所有门，把问题一次暴露给用户（最后统一汇总）；
#  * 依赖顺序固定：策略/单测 → 生成物漂移 → 类型检查 → TS 测试 → 断言报告 → 对拍 → 视觉 → 书 → 进度；
#  * 每一步的 stdout/stderr 都落盘到 dist/logs/，失败时打印末尾 20 行（不用翻屏找错误）；
#  * 无 xelatex / 无浏览器等环境缺失只降级、不假装通过：报告里写明 "skipped"。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/.cache/mpl}"
export npm_config_cache="${npm_config_cache:-$ROOT/.cache/npm}"
mkdir -p dist/logs dist/reports .cache/mpl .cache/npm

PY="${PYTHON:-python3}"
FAILED=0
declare -a NAMES=() STATUS=() SECONDS_ARR=()

run_gate() {
  local id="$1" name="$2" cmd="$3"
  local log="dist/logs/${id}.log"
  local t0 t1 rc
  t0=$(date +%s.%N)
  printf '\n\033[1m=== %s %s ===\033[0m\n$ %s\n' "$id" "$name" "$cmd"
  if bash -lc "$cmd" >"$log" 2>&1; then
    rc=0
  else
    rc=$?
  fi
  t1=$(date +%s.%N)
  local secs
  secs=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')
  NAMES+=("$id $name"); SECONDS_ARR+=("$secs")
  if [[ $rc -eq 0 ]]; then
    STATUS+=("pass")
    printf '  ✅ 通过（%s s）\n' "$secs"
  elif [[ $rc -eq 77 ]]; then
    STATUS+=("skipped")
    printf '  ⚠️  跳过（环境不支持；%s s）\n' "$secs"
  else
    STATUS+=("fail")
    FAILED=$((FAILED + 1))
    printf '  ❌ 失败（exit=%s，%s s）—— 末尾 20 行：\n' "$rc" "$secs"
    tail -n 20 "$log" | sed 's/^/     /'
  fi
}

echo "仓库：$ROOT"
echo "python：$($PY -V 2>&1)　node：$(node -v 2>/dev/null || echo 缺失)"

run_gate G1 "策略红线（POLICY P1–P13 中的 11 条机器规则）" "$PY tools/check_policy.py --quiet"
run_gate G2 "Python 测试（pytest，严格模式）"  "$PY -m pytest tests -q"
run_gate G6 "生成物无漂移（manifest + 色标）"   "$PY tools/gen_manifests.py --check && $PY tools/gen_colorscales.py --check"
run_gate G4 "TypeScript 类型检查"             "npm run typecheck"
run_gate G5 "TypeScript 测试"                 "npm run build --silent && npm test"
# G3 复用 G2 已经写好的报告：不重复跑一遍 pytest
run_gate G3 "数值断言对账（声明↔实测）"        "$PY tools/run_asserts.py --no-run"
run_gate G7 "Python ↔ TS 对拍"                "$PY tools/parity.py"
# G8 的前置：先算 L3 帧（--n 6，与现有基线的场景数**必须一致**），再用真前端模型转储场景，
# 最后才比对基线。少了前两步，新检出里 dist/scenes 为空 → 全部走兜底 → "一张图都没比"也能绿。
run_gate G8 "视觉回归（帧→TS 场景→栅格基线）"  "$PY tools/precompute.py --n 6 --quiet && npm run scene --silent && $PY tools/visual_regression.py --check --require-scenes"
run_gate G9 "内容体检 + LaTeX（逐章 + 合并本）与站点公式"  "$PY tools/check_chapter_content.py --all && $PY tools/build_book.py --no-compile && $PY tools/build_book.py --check && $PY tools/check_site_formulas.py"
run_gate G10 "进度文档无漂移"                  "$PY tools/gen_progress.py --check"

# --- 汇总 ---
{
  echo "{"
  echo "  \"generated_at\": \"$(date -Iseconds)\","
  echo "  \"commit\": \"$(git rev-parse --short HEAD 2>/dev/null || echo unknown)\","
  echo "  \"failed\": $FAILED,"
  echo "  \"ok\": $([ $FAILED -eq 0 ] && echo true || echo false),"
  echo "  \"gates\": ["
  for i in "${!NAMES[@]}"; do
    id="${NAMES[$i]%% *}"
    comma=","; [[ $i -eq $((${#NAMES[@]} - 1)) ]] && comma=""
    echo "    {\"id\": \"$id\", \"name\": \"${NAMES[$i]#* }\", \"status\": \"${STATUS[$i]}\", \"seconds\": ${SECONDS_ARR[$i]}, \"log\": \"dist/logs/$id.log\"}$comma"
  done
  echo "  ]"
  echo "}"
} > dist/reports/ci.json

printf '\n\033[1m=== 汇总 ===\033[0m\n'
for i in "${!NAMES[@]}"; do
  case "${STATUS[$i]}" in
    pass) mark="✅" ;;
    skipped) mark="⚠️ " ;;
    *) mark="❌" ;;
  esac
  printf '  %s %-46s %6s s\n' "$mark" "${NAMES[$i]}" "${SECONDS_ARR[$i]}"
done
if [[ $FAILED -eq 0 ]]; then
  printf '\n\033[32m全部门禁通过。\033[0m 报告：dist/reports/ci.json\n'
  exit 0
fi
printf '\n\033[31m%d 道门禁失败。\033[0m 逐个看：dist/logs/<门禁>.log；汇总：dist/reports/ci.json\n' "$FAILED"
exit 1
