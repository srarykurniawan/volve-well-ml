.PHONY: install train train-fast test api dashboard docker

install:
	pip install -e ".[api,dashboard,dev]"

train:
	python -m wellml.pipeline

train-fast:
	python -m wellml.pipeline --fast

test:
	pytest -q

api:
	uvicorn api.app:app --reload --port 8000

dashboard:
	streamlit run dashboard/streamlit_app.py

docker:
	docker build -t wellml:latest .
