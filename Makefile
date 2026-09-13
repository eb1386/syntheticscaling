# Single-RTX-5080 study — common commands.
PROFILE ?= scaling5080
PYTHON  ?= python3

.PHONY: help install budget smoke micro run run-fast test clean

help:
	@echo "make install   PROFILE=$(PROFILE)   # deps + data + models + pool (one time)"
	@echo "make budget    PROFILE=$(PROFILE)   # print GPU-hour / wall-clock estimate"
	@echo "make micro                          # CPU end-to-end validation (minutes)"
	@echo "make smoke                          # GPU end-to-end validation (~1h, real models, tiny budget)"
	@echo "make run       PROFILE=$(PROFILE)   # run the full study"
	@echo "make run-fast                       # run the ~1-week reduced study"
	@echo "make test                           # unit tests"

install:
	./install.sh $(PROFILE)

budget:
	$(PYTHON) -m synscale.analysis.compute_budget --profile $(PROFILE)

micro:
	SYNSCALE_STORE=checkpoints/micro $(PYTHON) -m synscale.pipeline --profile micro --device cpu --max-steps 40 --eval-limit 20

smoke:
	./run_all.sh smoke --eval-limit 200

run:
	./run_all.sh $(PROFILE)

run-fast:
	./run_all.sh local5080_fast

test:
	$(PYTHON) -m pytest -q

clean:
	rm -rf results/micro checkpoints/micro data/processed/micro data/prompts/pool-micro.jsonl configs/experiments/micro
