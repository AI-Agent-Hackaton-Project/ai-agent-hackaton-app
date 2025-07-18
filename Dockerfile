FROM python:3.13-slim

# 環境変数を設定
ENV PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.2 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    graphviz \
    libgraphviz-dev \
    && curl -sSL https://install.python-poetry.org | python - \
    && apt-get purge -y --auto-remove curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry
    
WORKDIR /app

COPY pyproject.toml poetry.lock* /app/

RUN poetry install --no-interaction --no-ansi

COPY . /app/

CMD ["poetry", "run", "streamlit", "run", "app/main.py", "--server.port=8080", "--server.address=0.0.0.0"]