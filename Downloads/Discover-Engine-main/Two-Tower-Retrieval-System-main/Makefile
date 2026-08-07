.PHONY: install data train index serve api test ablation clean docker-build docker-run

install:
	pip install -r requirements.txt
	pip install pytest

data:
	mkdir -p data/raw
	cd data/raw && curl -sSL -o ml-1m.zip https://files.grouplens.org/datasets/movielens/ml-1m.zip && unzip -q -o ml-1m.zip

train:
	python train.py

index:
	python build_index.py

serve:
	python serve.py

api:
	uvicorn api.app:app --reload --port 8000

test:
	KMP_DUPLICATE_LIB_OK=TRUE pytest tests/ -v

ablation:
	python ablation.py

compare:
	python compare.py

api-bpr:
	LOSS=bpr uvicorn api.app:app --reload --port 8000

api-infonce:
	LOSS=infonce uvicorn api.app:app --reload --port 8001

clean:
	rm -rf artifacts/*.pt artifacts/*.pkl artifacts/*.index __pycache__ .pytest_cache

docker-build:
	docker build -t two-tower-retrieval .

docker-run:
	docker run -p 8000:8000 -v $(PWD)/artifacts:/app/artifacts two-tower-retrieval
