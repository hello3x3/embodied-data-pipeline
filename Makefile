PY ?= python3
OUT ?= outputs/data_pipeline_web
TASKS ?= fill_pen_holder stack_bowls insert_tubes
EPISODE ?= episode_0000000.hdf5
OBJECT ?= sam2
ARGS ?=

.PHONY: help build page diag-object diag-samples diag-tasks check clean

help:
	@echo "make build         build sidecar + review page"
	@echo "make page          serve $(OUT) on http://127.0.0.1:8772"
	@echo "make check         compile + unit tests"
	@echo "make diag-object   TASK=... SUB=...  (8-panel diagnostic)"
	@echo "make diag-samples  10 subtask samples"
	@echo "make diag-tasks    10 task samples"

build:
	MPLCONFIGDIR=$${MPLCONFIGDIR:-/tmp/mplcache} $(PY) -m data_pipeline.build_plan_queue \
		--tasks $(TASKS) --episode $(EPISODE) --object $(OBJECT) \
		--out $(OUT) $(ARGS)

page:
	$(PY) -m http.server 8772 --bind 127.0.0.1 --directory $(OUT)

TASK ?= fill_pen_holder
SUB ?= 0
diag-object:
	MPLCONFIGDIR=$${MPLCONFIGDIR:-/tmp/mplcache} $(PY) -m data_pipeline.diag_object \
		--task $(TASK) --sub $(SUB) --out outputs/diag/$(TASK)_s$(SUB)

diag-samples:
	$(PY) -m data_pipeline.diag_samples

diag-tasks:
	$(PY) -m data_pipeline.diag_tasks

check:
	$(PY) -m compileall -q data_pipeline
	$(PY) -m unittest discover -s tests -v

clean:
	rm -rf outputs .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
