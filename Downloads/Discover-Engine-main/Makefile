.PHONY: install data train index api test verify ablation clean

PYTHON ?= python
PIP ?= pip

install:
	$(PIP) install -r requirements.txt

data:
	$(PYTHON) training/prepare_dataset.py

train:
	$(PYTHON) training/train_user_tower.py --epochs 5

index:
	$(PYTHON) scripts/build_index.py

api:
	$(PYTHON) -m uvicorn serving.api:app --reload --host 0.0.0.0 --port 8000

test:
	set KMP_DUPLICATE_LIB_OK=TRUE && $(PYTHON) -m pytest tests/ -v

verify:
	$(PYTHON) scripts/verify_embeddings.py

ablation:
	$(PYTHON) eval/ablation.py

clean:
	-$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('artifacts').glob('*')]"

# Full pipeline
all: install data train index
