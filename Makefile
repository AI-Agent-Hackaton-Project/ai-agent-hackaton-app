COMPOSE = docker-compose
VENV_DIR = venv

.PHONY: up build down restart logs venv start

## コンテナをビルドしてバックグラウンドで起動する
up:
	$(COMPOSE) up -d

## Docker イメージをビルドする
build:
	$(MAKE) up
	$(COMPOSE) build
	$(MAKE) venv

## コンテナとネットワークを停止・削除する
down:
	$(COMPOSE) down

## コンテナを再起動する（down → up）
restart:
	$(MAKE) down
	$(MAKE) up

## Streamlit アプリケーションを実行する
start:
	$(COMPOSE) run --rm app streamlit run app/main.py & sleep 2 && open http://localhost:8501

## 仮想環境の作成とパッケージのインストール
venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		python3 -m venv $(VENV_DIR); \
	fi
	$(VENV_DIR)/bin/pip install --upgrade pip
