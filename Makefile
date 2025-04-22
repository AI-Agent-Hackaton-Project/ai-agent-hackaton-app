# 使用する docker-compose コマンド
COMPOSE=docker-compose

.PHONY: up build down restart logs

## コンテナをビルドしてバックグラウンドで起動する
up:
	$(COMPOSE) up -d

## Docker イメージをビルドする
build:
	$(COMPOSE) build

## コンテナとネットワークを停止・削除する
down:
	$(COMPOSE) down

## コンテナを再起動する（down → up）
restart:
	$(MAKE) down
	$(MAKE) up

## ログをリアルタイムで表示する
logs:
	$(COMPOSE) logs -f
