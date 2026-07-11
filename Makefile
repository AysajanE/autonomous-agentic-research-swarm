.PHONY: gate reproduce-analysis paper

gate:
	python scripts/quality_gates.py

reproduce-analysis:
	python scripts/reproduce_analysis.py

paper:
	python scripts/render_paper.py

.PHONY: test

test:
	python -m unittest discover -s tests

.PHONY: swarm-plan

swarm-plan:
	python scripts/swarm.py plan

.PHONY: swarm-tick

swarm-tick:
	python scripts/swarm.py tick

.PHONY: sweep

sweep:
	python scripts/sweep_tasks.py
