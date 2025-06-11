import os
import tempfile
from typing import TypedDict, List, Dict, Any, Iterator

from config.env_config import get_env_config
from .workflow_steps import (
    generate_search_query,
    perform_google_search,
    scrape_and_prepare_context,
    generate_article_content,
    generate_aphorism,
    generate_main_image,
    format_html,
)


# 状態管理用のクラス
class AgentState(TypedDict):
    main_title: str
    subtitles: List[str]
    search_query: str
    raw_search_results: List[Dict[str, Any]]
    scraped_context: str
    generated_article_json: Dict[str, Any]
    initial_article_title: str
    initial_article_content: str
    main_theme_image_path: str | None
    subtitle_image_paths: List[str] | None
    aphorism: str
    html_output: str
    error: str | None


def generate_single_subtitle_image(
    llm,
    image_model,
    main_title: str,
    subtitle: str,
    regional_characteristics: str,
    temp_dir: str,
    index: int,
) -> str | None:
    """単一サブタイトル画像の生成"""
    try:
        from utils.generate_titles_images import _generate_image_prompt, _generate_image

        # 画像プロンプト生成
        image_prompt = _generate_image_prompt(
            llm, main_title, main_title, subtitle, regional_characteristics
        )

        # 画像生成実行
        image_bytes = _generate_image(image_model, image_prompt, index, 1)

        if image_bytes:
            # ファイル名の安全化
            safe_subtitle = "".join(
                c if c.isalnum() or c in "-_" else "_" for c in subtitle
            )[:50]
            file_name = f"image_{index+1:02d}_{safe_subtitle}.png"
            image_file_path = os.path.join(temp_dir, file_name)

            # 画像ファイル保存
            with open(image_file_path, "wb") as f:
                f.write(image_bytes)

            (f"💾 画像 {index + 1} を保存: {image_file_path}")
            return image_file_path
        return None

    except Exception as e:
        (f"画像生成エラー: {e}")
        return None


def generate_article_workflow(
    main_title_input: str,
    subtitles_input: List[str],
    attempt_prefecture_image: bool = True,
) -> Iterator[Dict[str, Any]]:
    """記事生成ワークフローのメイン関数"""

    (f"\n--- 「{main_title_input}」に関する記事生成を開始します ---")

    # 初期状態設定
    state = AgentState(
        main_title=main_title_input,
        subtitles=subtitles_input,
        search_query="",
        raw_search_results=[],
        scraped_context="",
        generated_article_json={},
        initial_article_title="",
        initial_article_content="",
        main_theme_image_path=None,
        subtitle_image_paths=None,
        aphorism="",
        html_output="",
        error=None,
    )

    try:
        # ステップ1: 検索クエリ生成
        yield {
            "step": "search_query",
            "message": "検索クエリを生成しています",
            "state": state,
        }
        state = generate_search_query(state)

        # ステップ2: Google検索実行
        yield {
            "step": "google_search",
            "message": "Web検索を実行しています",
            "state": state,
        }
        state = perform_google_search(state)

        # ステップ3: コンテキスト情報収集
        yield {
            "step": "scrape_context",
            "message": "関連情報を収集しています",
            "state": state,
        }
        state = scrape_and_prepare_context(state)

        # ステップ4: 記事本文生成
        yield {
            "step": "generate_article",
            "message": "記事本文を生成しています",
            "state": state,
        }
        state = generate_article_content(state)

        # ステップ5: 地域の名言生成
        yield {
            "step": "generate_aphorism",
            "message": "地域の名言を生成しています",
            "state": state,
        }
        state = generate_aphorism(state)

        # ステップ6: 4コマ画像生成
        if attempt_prefecture_image:
            yield {
                "step": "main_image",
                "message": "4コマ画像を生成しています",
                "state": state,
                "image_progress": {"type": "main_image"},
            }
            state = generate_main_image(state, attempt_prefecture_image)

        # ステップ7以降: サブタイトル画像の個別生成
        if attempt_prefecture_image and state["subtitles"]:
            total_count = len(state["subtitles"])

            try:
                # 設定読み込み
                settings = get_env_config()

                # 画像生成モデル初期化
                from utils.generate_titles_images import (
                    _initialize_vertex_ai,
                    _generate_regional_characteristics,
                )

                image_model, llm = _initialize_vertex_ai(
                    settings["gcp_project_id"],
                    settings["gcp_location"],
                    settings.get(
                        "image_gen_model_name", "imagen-3.0-fast-generate-001"
                    ),
                    settings.get("model_name", "gemini-1.5-pro-001"),
                )

                if not image_model or not llm:
                    raise ValueError("モデル初期化に失敗しました")

                # 地域特性生成（一度だけ）
                regional_characteristics = _generate_regional_characteristics(
                    llm, state["main_title"]
                )

                # 一時ディレクトリ作成
                temp_dir = tempfile.mkdtemp(prefix=f"img_{state['main_title']}_")
                generated_paths = []

                # 各サブタイトル画像を順次生成
                for i, subtitle in enumerate(state["subtitles"]):
                    # 進捗状況をUI送信（画像生成開始前）
                    yield {
                        "step": f"subtitle_image_{i+1}",
                        "message": f"サブ画像生成中 [{i+1}/{total_count}]: 「{subtitle}」",
                        "state": state,
                        "image_progress": {
                            "type": "subtitle_image",
                            "current": i + 1,
                            "total": total_count,
                            "subtitle": subtitle,
                        },
                    }

                    # 実際の画像生成処理
                    image_path = generate_single_subtitle_image(
                        llm,
                        image_model,
                        state["main_title"],
                        subtitle,
                        regional_characteristics,
                        temp_dir,
                        i,
                    )

                    # 生成成功時はパスを保存
                    if image_path:
                        generated_paths.append(image_path)

                # 生成されたパスを状態に保存
                state["subtitle_image_paths"] = generated_paths

            except Exception as e:
                state["error"] = f"サブ画像生成エラー: {e}"
                (f"サブ画像生成エラー: {e}")

        # 最終ステップ: HTML整形
        yield {"step": "format_html", "message": "HTMLを整形しています", "state": state}
        state = format_html(state)

        # 完了通知
        yield {"step": "__end__", "message": "記事生成が完了しました", "state": state}

    except Exception as e:
        # エラー発生時の処理
        yield {
            "step": "workflow_error",
            "error": f"ワークフロー実行エラー: {e}",
            "message": f"ワークフロー実行中にエラーが発生しました: {e}",
            "state": state,
        }
