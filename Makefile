# dreamforge -- convenience targets
# Usage: make test | make mine | make imagine | make verify | make report | make pilot
RUN ?= runs/main
TASK ?= libero_spatial
EPISODES ?= 20
WM ?= aso
PY ?= python

.PHONY: test mine imagine verify build train eval report dashboard diagnose figures pilot clean

test:
	$(PY) -m pytest tests -q

dashboard:
	$(PY) -m dreamforge.dashboard --run $(RUN) --out results/dashboard.html

diagnose:
	$(PY) -m dreamforge.diagnose --mine $(RUN)/mine --wms base aso full \
		--out $(RUN)/wm_diagnosis.json

mine:
	$(PY) -m dreamforge.mine --task $(TASK) --episodes $(EPISODES) --out $(RUN)/mine

imagine:
	$(PY) -m dreamforge.imagine --mine $(RUN)/mine --out $(RUN)/imagine --wm $(WM) --context failure

verify:
	$(PY) -m dreamforge.verify --mine $(RUN)/mine --imagine $(RUN)/imagine \
		--out $(RUN)/verify --arms none imagined random --exec-horizon 16

build:
	$(PY) -m dreamforge.build --mine $(RUN)/mine --imagine $(RUN)/imagine \
		--out $(RUN)/build --filter all

train:
	$(PY) -m dreamforge.train --samples $(RUN)/build --out $(RUN)/ft_synth \
		--steps 3000 --replay-ratio 0.5

eval:
	$(PY) -m dreamforge.eval --policy-path $(RUN)/ft_synth --task $(TASK) \
		--episodes 10 --out $(RUN)/eval_ft.json

report:
	$(PY) -m dreamforge.report --run $(RUN)

figures:
	$(PY) -m dreamforge.figures --run $(RUN)

pilot:
	$(PY) -m dreamforge.mine --task libero_spatial --task-ids 0 --episodes 2 --out $(RUN)/mine
	$(PY) -m dreamforge.imagine --mine $(RUN)/mine --out $(RUN)/imagine --wm $(WM) --context all
	$(PY) -m dreamforge.verify --mine $(RUN)/mine --imagine $(RUN)/imagine \
		--out $(RUN)/verify --arms none imagined random --exec-horizon 16

clean:
	rm -rf runs/pilot
