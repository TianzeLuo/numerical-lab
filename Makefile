# 交互式数值方法实验室 —— 常用入口。全部命令都能在仓库根直接跑。
#
# 约定：所有"生成物"都写进 dist/ 或 data/；所有"契约"（manifest、golden、基线）进 git。

PY      ?= python3
NPM     ?= npm
PORT    ?= 8080
CHAPTER ?= 0.0

.DEFAULT_GOAL := help
.PHONY: help ci setup test pytest jstest typecheck build policy asserts parity visual manifests book progress serve precompute clean clean-all

help: ## 显示这份帮助
	@grep -hE "^[a-zA-Z_-]+:.*?## " $(MAKEFILE_LIST) | awk "BEGIN{FS=\":.*?## \"}{printf \"  \\033[36m%-14s\\033[0m %s\\n\", \$$1, \$$2}"

ci: ## 跑完 G1–G10 十道门禁（失败不早停，逐个汇总）
	bash tools/ci.sh

setup: ## 检查环境（python/node/依赖/字体），不装任何东西
	$(PY) -c "import sys, numpy; print(\"python\", sys.version.split()[0], \"numpy\", numpy.__version__)"
	@node -v && $(NPM) -v
	@command -v xelatex >/dev/null && xelatex --version | head -1 || echo "（无 xelatex：G9 会降级为结构检查）"

test: pytest jstest ## Python + TypeScript 全部测试

pytest: ## Python 测试
	$(PY) -m pytest tests -q

jstest: build ## TypeScript 测试（依赖 dist/，所以先编译）
	$(NPM) test

typecheck: ## tsc --noEmit
	$(NPM) run typecheck

build: ## tsc 编译到 dist/viz
	$(NPM) run build --silent

policy: ## G1 策略红线静态检查
	$(PY) tools/check_policy.py

asserts: ## G3 跑断言并生成报告/章末附表
	$(PY) tools/run_asserts.py

parity: ## G7 Python ↔ TS 对拍
	$(PY) tools/parity.py

visual: ## G8 视觉回归（--check）；更新基线用 make visual-update
	$(PY) tools/visual_regression.py --check

visual-update: ## G8 显式更新视觉基线（必须在提交信息里写明原因）
	$(PY) tools/visual_regression.py --update

manifests: ## G6 重新生成 manifest 产物
	$(PY) tools/gen_manifests.py

book: ## G9 逐章 + 合并本 LaTeX
	$(PY) tools/build_book.py

progress: ## G10 重新生成 docs/PROGRESS.md
	$(PY) tools/gen_progress.py

serve: ## 起本地服务（HTTP + WebSocket）
	$(PY) -m server.app --port $(PORT) --allow-write

precompute: ## 生成 L3 静态帧到 data/frames/
	$(PY) tools/precompute.py --chapter $(CHAPTER)

clean: ## 删掉 dist/ 与缓存（保留 data/ 契约产物）
	rm -rf dist .cache/pytest .cache/mpl

clean-all: clean ## 连 data/frames 一起删（可重新生成）
	rm -rf data/frames
