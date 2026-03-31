FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agents/ ./agents/
COPY tools/ ./tools/
COPY langflow_components/ ./langflow_components/
COPY data/ ./data/
COPY rag_storage/ ./rag_storage/
COPY run_agents.py .

RUN mkdir -p /app/workspace

CMD ["python", "run_agents.py", "--data-path", "./data/train.csv", "--test-path", "./data/test.csv", "--target-column", "target"]
