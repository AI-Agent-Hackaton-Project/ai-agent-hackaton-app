# AI Agent Hackaton Project


Docker Compose を使用して、開発環境の構築とアプリケーションの実行を行います。

## 必要なもの (Prerequisites)

* [Docker](https://www.docker.com/)
* [Docker Compose](https://docs.docker.com/compose/install/) (Docker Desktop には通常含まれています)

お使いの環境に上記がインストールされていることを確認してください。

## 実行手順 (Usage)

1.  **Docker イメージのビルド:**
    最初に、アプリケーションの実行に必要な Docker イメージをビルドします。
    プロジェクトのルートディレクトリ（`docker-compose.yml` がある場所）で、以下のコマンドを実行してください。

    ```bash
    docker-compose build
    ```

2.  **アプリケーションの起動:**
    イメージのビルドが完了したら、以下のコマンドでコンテナを起動します。

    ```bash
    docker-compose up
    ```

    * このコマンドを実行すると、ログがターミナルに表示され続けます (フォアグラウンド実行)。
    * ターミナルを占有せず、バックグラウンドで実行したい場合は `-d` オプションを使用します: `docker-compose up -d`

3.  **アプリケーションへのアクセス:**
    コンテナの起動後、Webブラウザを開き、以下のアドレスにアクセスしてください。

    * `http://localhost:8501`

    Streamlitアプリケーションが表示されるはずです。

## アプリケーションの停止 (Stopping)

* **フォアグラウンド実行 (`docker-compose up`) した場合:**
    * コマンドを実行したターミナルで `Ctrl + C` を押すと停止します。

* **バックグラウンド実行 (`docker-compose up -d`) した場合、または完全にコンテナを削除したい場合:**
    * 以下のコマンドを実行します。これにより、コンテナが停止され、削除されます。

    ```bash
    docker-compose down
    ```

## [その他 (Optional)]

[もしあれば、開発時の注意点や補足情報などを記載します。 例: ホットリロードが有効になっています、など]

---