FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[api]"
COPY config ./config
COPY api ./api
COPY data/raw ./data/raw
# latih model saat build (atau mount folder models/ hasil training)
RUN python -m wellml.pipeline --fast
EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
