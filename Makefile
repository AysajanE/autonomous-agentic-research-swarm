PYTHON ?= python3.11

.PHONY: gate reproduce-analysis paper drill eval-heldout bt2a-rehearsal

gate:
	$(PYTHON) scripts/quality_gates.py

bt2a-rehearsal:
	$(PYTHON) scripts/bt2a_rehearsal.py

reproduce-analysis:
	$(PYTHON) scripts/reproduce_analysis.py

paper:
	$(PYTHON) scripts/render_paper.py

drill:
	$(PYTHON) scripts/seeded_drill.py --all

eval-heldout:
	$(PYTHON) -m unittest discover -s tests/held_out -p 'cases.py'

.PHONY: test

test:
	$(PYTHON) -m unittest discover -s tests

.PHONY: swarm-plan

swarm-plan:
	$(PYTHON) scripts/swarm.py plan

.PHONY: swarm-tick

swarm-tick:
	$(PYTHON) scripts/swarm.py tick

.PHONY: sweep

sweep:
	$(PYTHON) scripts/sweep_tasks.py

.PHONY: swarm-init

swarm-init:
	$(PYTHON) scripts/swarm_init.py --mode $(MODE) --output $(OUTPUT)
