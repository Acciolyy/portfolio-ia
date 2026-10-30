# Re-run the experiments from an activated virtualenv, or pass an absolute path:
#   make PYTHON=$(pwd)/.venv/bin/python <target>   (a relative path breaks after cd)
PYTHON ?= python
P2 = portfolio-2-busca-otimizacao
P3 = portfolio-3-modelos-probabilisticos
P4 = portfolio-4-machine-learning

.PHONY: all p2 p3 p4 seeds test lint

# Default runs (one seed). The scripts write figures to the current directory,
# so each one runs from the folder where its figures are kept.
all: p2 p3 p4

p2:
	cd $(P2)/src && for f in projeto_01_busca_cega informed_search tsp_local_search n_queens_genetic csp_map_coloring wumpus_world; do $(PYTHON) $$f.py; done

p3:
	cd $(P3)/imagens/projeto-1-rede-bayesiana && $(PYTHON) ../../src/projeto_01_rede_bayesiana.py
	cd $(P3)/imagens/projeto-2-hmm && $(PYTHON) ../../src/projeto_02_hmm.py
	cd $(P3)/imagens/projeto-3-kalman && $(PYTHON) ../../src/projeto_03_kalman.py

p4:
	cd $(P4) && for f in projeto_01_supervisionado projeto_02_nao_supervisionado projeto_03_reforco projeto_04_deep_learning; do $(PYTHON) src/$$f.py; done

# The multi-seed / repeated-timing evaluations whose numbers the READMEs report.
seeds:
	cd $(P2)/src && $(PYTHON) projeto_01_busca_cega.py --repeat 2000 && $(PYTHON) informed_search.py --repeat 2000 \
		&& $(PYTHON) csp_map_coloring.py --repeat 2000 && $(PYTHON) tsp_local_search.py --seeds 30 \
		&& $(PYTHON) n_queens_genetic.py --seeds 30
	cd $(P3)/src && $(PYTHON) projeto_02_hmm.py --seeds 30 && $(PYTHON) projeto_03_kalman.py --seeds 30
	cd $(P4) && $(PYTHON) src/projeto_03_reforco.py --seeds 10 && $(PYTHON) src/projeto_04_deep_learning.py --seeds 5

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .
