COMPOSE = docker-compose
VENV_DIR = venv

.PHONY: up build down restart logs venv start requirements-update

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

## requirements.txtを更新し、変更があった場合のみコミット/プッシュ
requirements-update:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		python3 -m venv $(VENV_DIR); \
	fi
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r requirements.txt
	. $(VENV_DIR)/bin/activate && pip freeze > requirements.txt
	@if git diff --quiet requirements.txt; then \
		echo "🟢 requirements.txt に変更はありません。コミットをスキップします。"; \
	else \
		echo "🔵 requirements.txt が変更されましたが、CI/CD でコミットとプッシュが行われます。"; \
	fi
