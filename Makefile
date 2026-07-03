install:
	pip install -r requirements.txt

db-init:
	alembic upgrade head

ingest:
	python data/pipeline.py

embed:
	python rag/ingest.py

train:
	python model/train.py

evaluate:
	python model/evaluate.py

serve:
	uvicorn api.main:app --reload --port 8000

test:
	python api/test_requests.py

lint:
	flake8 . --max-line-length=100 --exclude=.venv,db/migrations,__pycache__

frontend:
	open http://localhost:8000/ui

all: install db-init ingest embed train serve
