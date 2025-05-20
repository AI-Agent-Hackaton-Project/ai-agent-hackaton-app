import os
import uuid  # uuidは現状コードでは直接使われていませんが、将来的な拡張やログのために残しておきます
import json
from typing import TypedDict, List, Dict, Any
import traceback  # エラーハンドリングのためインポート

from langchain_google_vertexai import ChatVertexAI

from langchain_core.messages import (
    HumanMessage,
)
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException

# Pydanticモデル
from pydantic import BaseModel, Field

from prompts.GENERATE_ARTICLE import (
    GENERATE_ARTICLE_PROMPT,
)
from config.env_config import get_env_config

# Google検索とWebスクレイピング用ツール
from langchain_google_community.search import GoogleSearchAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_transformers import BeautifulSoupTransformer

# LangGraph
from langgraph.graph import StateGraph, END


# --- Pydanticモデル定義 (記事構造) ---
class Article(BaseModel):
    title: str = Field(description="記事のタイトル")
    block: List[str] = Field(description="記事の各ブロックの本文リスト")


# --- 状態の定義 ---
class AgentState(TypedDict):
    topic: str  # 記事のトピック
    search_query: str  # 生成された検索クエリ
    raw_search_results: List[Dict[str, Any]]  # Google検索の生の結果
    scraped_context: str  # スクレイピング・整形されたコンテキスト
    generated_article_json: Dict[str, Any]  # LLMによって生成された記事のJSON表現
    initial_article_title: str  # 初期生成された記事のタイトル
    initial_article_content: str  # 初期生成された記事の本文 (ブロックを結合)
    revised_article: str  # 添削された記事
    html_output: str  # 最終的なHTML出力
    error: str | None  # エラーメッセージ


def generate_article_workflow(
    topic_input: str, output_dir: str = "."
) -> Dict[str, Any]:
    """
    指定されたトピックに基づいて記事を生成し、HTMLファイルとして保存するワークフローを実行します。

    Args:
        topic_input (str): 記事を生成するトピック。
        output_dir (str, optional): 生成されたHTMLファイルを保存するディレクトリ。
                                    デフォルトはカレントディレクトリ。

    Returns:
        Dict[str, Any]: ワークフローの実行結果。以下のキーを含む可能性があります:
            - "success" (bool): 処理が成功したかどうか。
            - "topic" (str): 入力されたトピック。
            - "output_file_path" (str | None): 生成されたHTMLファイルのパス。エラー時はNone。
            - "error_message" (str | None): エラーが発生した場合のメッセージ。成功時はNone。
            - "final_state_summary" (Dict | None): 最終状態の主要な情報の要約（デバッグ用）。
    """
    print(f"\n--- 「{topic_input}」に関する記事生成を開始します ---")
    print(f"--- 出力先ディレクトリ: {os.path.abspath(output_dir)} ---")

    # --- 設定の読み込み ---
    settings = get_env_config()

    # --- 必須設定のチェック ---
    google_api_key = settings.get("google_api_key")
    google_cse_id = settings.get("google_cse_id")

    if not google_api_key or not google_cse_id:
        error_msg = "エラー: Google APIキーまたはCSE IDが設定ファイルに存在しません。"
        print(error_msg)
        return {
            "success": False,
            "topic": topic_input,
            "output_file_path": None,
            "error_message": error_msg,
            "final_state_summary": None,
        }

    # --- LLMとツールの初期化 ---
    try:
        llm = ChatVertexAI(
            model_name=settings.get("model_name", "gemini-1.0-pro-001"),
            temperature=0,
            max_output_tokens=settings.get("max_output_tokens", 8192),
            max_retries=6,
            stop=None,
        )
        search_wrapper = GoogleSearchAPIWrapper(
            google_api_key=google_api_key,
            google_cse_id=google_cse_id,
        )
        output_parser = PydanticOutputParser(pydantic_object=Article)
    except Exception as e_init:
        error_msg = f"LLMまたはツールの初期化中にエラーが発生しました: {e_init}"
        print(error_msg)
        traceback.print_exc()
        return {
            "success": False,
            "topic": topic_input,
            "output_file_path": None,
            "error_message": error_msg,
            "final_state_summary": None,
        }

    # --- ノード定義 (内部関数として定義) ---
    def generate_search_query_node(state: AgentState) -> AgentState:
        print("--- ステップ1a: 検索クエリ生成 ---")
        topic = state["topic"]
        search_query = f"{topic} 歴史 最新情報 観光"
        print(f"生成された検索クエリ: {search_query}")
        return {**state, "search_query": search_query, "error": None}

    def google_search_node(state: AgentState) -> AgentState:
        if state.get("error"):
            return state
        print("--- ステップ1b: Google検索実行 ---")
        query = state["search_query"]
        num_results = settings.get("num_search_results", 3)
        try:
            print(f"Google検索中 (クエリ: {query}, {num_results}件)...")
            search_results_list = search_wrapper.results(
                query=query, num_results=num_results
            )
            print(
                f"検索結果 {len(search_results_list) if search_results_list else 0} 件取得完了。"
            )
            return {
                **state,
                "raw_search_results": (
                    search_results_list if search_results_list else []
                ),
                "error": None,
            }
        except Exception as e:
            print(f"Google検索中にエラーが発生しました: {e}")
            return {**state, "error": f"Google検索エラー: {str(e)}"}

    def scrape_and_prepare_context_node(state: AgentState) -> AgentState:
        if state.get("error"):
            return state
        print("--- ステップ1c: Webスクレイピングとコンテキスト準備 ---")
        search_results_list = state["raw_search_results"]
        scraped_contents = []
        max_content_length_per_page = settings.get(
            "max_content_length_per_page_scrape", 1500
        )

        if not search_results_list:
            return {
                **state,
                "scraped_context": "関連情報は見つかりませんでした。",
                "error": None,
            }

        print(
            f"検索結果から上位{len(search_results_list)}件のウェブページを読み込んでいます..."
        )
        for i, result in enumerate(search_results_list):
            title = result.get("title", "タイトルなし")
            link = result.get("link")
            snippet = result.get("snippet", "スニペットなし")
            if link:
                try:
                    print(f"  読み込み中 ({i+1}/{len(search_results_list)}): {link}")
                    loader = WebBaseLoader(
                        web_path=link, requests_kwargs={"timeout": 10}
                    )
                    documents = loader.load()
                    if documents:
                        bs_transformer = BeautifulSoupTransformer()
                        docs_transformed = bs_transformer.transform_documents(
                            documents,
                            tags_to_extract=[
                                "p",
                                "h1",
                                "h2",
                                "h3",
                                "li",
                                "span",
                                "article",
                            ],
                        )
                        page_content = " ".join(
                            [doc.page_content for doc in docs_transformed]
                        )
                        shortened_content = page_content[
                            :max_content_length_per_page
                        ].strip()
                        if shortened_content:
                            scraped_contents.append(
                                f"参照元URL: {link}\nタイトル: {title}\n内容:\n{shortened_content}"
                            )
                            print(
                                f"    -> コンテンツ取得成功 (先頭{len(shortened_content)}文字)"
                            )
                        else:
                            scraped_contents.append(
                                f"参照元URL: {link}\nタイトル: {title}\n概要: {snippet}\n(主要コンテンツ抽出失敗)"
                            )
                            print(f"    -> 主要コンテンツ抽出失敗、スニペット利用")
                    else:
                        scraped_contents.append(
                            f"参照元URL: {link}\nタイトル: {title}\n概要: {snippet}\n(コンテンツ取得失敗: ドキュメントなし)"
                        )
                        print(f"    -> コンテンツ取得失敗 (ドキュメントなし)")
                except Exception as e_scrape:
                    print(
                        f"    -> URLからのコンテンツ読み込みエラー: {link}, エラー: {e_scrape}"
                    )
                    scraped_contents.append(
                        f"参照元URL: {link}\nタイトル: {title}\n概要: {snippet}\n(コンテンツの読み込みに失敗しました: {type(e_scrape).__name__})"
                    )
            else:
                scraped_contents.append(f"タイトル: {title}\n概要: {snippet} (URLなし)")

        search_context_str = (
            "\n\n===\n\n".join(scraped_contents)
            if scraped_contents
            else "関連性の高いウェブページのコンテンツは見つかりませんでした。"
        )
        print("ウェブページ読み込み完了。")
        return {**state, "scraped_context": search_context_str, "error": None}

    def generate_structured_article_node(state: AgentState) -> AgentState:
        if state.get("error"):
            return state
        print("--- ステップ2: 構造化記事生成 ---")
        topic = state["topic"]
        search_context = state["scraped_context"]
        try:
            system_template = "あなたはプロのWEBライターです。提供された検索結果(ウェブページからの抜粋情報)と指示に基づいて、魅力的で正確な情報に基づいた記事を指定されたJSON形式で作成してください。"
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_template),
                    ("user", GENERATE_ARTICLE_PROMPT),
                ]
            ).partial(format_instructions=output_parser.get_format_instructions())
            chain = prompt | llm | output_parser
            print("LLMによる記事生成中...")
            input_data = {"topic": topic, "search_results": search_context}
            parsed_article_obj: Article = chain.invoke(input_data)
            generated_article_json = parsed_article_obj.model_dump()
            article_title = generated_article_json.get("title", "タイトルなし")
            article_blocks = generated_article_json.get("block", [])
            article_content = "\n\n".join(article_blocks)
            print(f"記事生成完了。タイトル: {article_title}")
            return {
                **state,
                "generated_article_json": generated_article_json,
                "initial_article_title": article_title,
                "initial_article_content": article_content,
                "error": None,
            }
        except OutputParserException as e_parse:
            llm_output_str = getattr(
                e_parse,
                "llm_output",
                str(e_parse.args[0] if e_parse.args else str(e_parse)),
            )
            print(
                f"記事生成中にOutputParserエラーが発生しました: {e_parse}\nLLM Raw Output:\n{llm_output_str}"
            )
            return {
                **state,
                "error": f"記事生成パーサーエラー: {str(e_parse)}\nLLM Output: {llm_output_str}",
            }
        except Exception as e:
            print(f"記事生成中に予期せぬエラーが発生しました: {e}")
            traceback.print_exc()
            return {**state, "error": f"記事生成エラー: {str(e)}"}

    def revise_article_node(state: AgentState) -> AgentState:
        if state.get("error"):
            return state
        print("--- ステップ3: 記事添削 ---")
        article_to_revise = state["initial_article_content"]
        if not article_to_revise:
            print("添削対象の記事がありません。スキップします。")
            return {
                **state,
                "revised_article": state.get("initial_article_content", ""),
                "error": None,
            }
        try:
            print("記事を添削します...")
            prompt_text = f"以下の日本語の記事を、より自然で読みやすく、誤字脱字や文法的な誤りがないように添削してください。\n記事の主要な内容は変えずに、表現を改善してください。\nですます調を維持してください。\n\n元の記事:\n{article_to_revise}\n\n添削後の記事:\n"
            response = llm.invoke([HumanMessage(content=prompt_text)])
            revised_article = response.content
            print(f"記事添削完了:\n{revised_article[:200]}...")
            return {**state, "revised_article": revised_article, "error": None}
        except Exception as e:
            print(f"記事添削中にエラーが発生しました: {e}")
            traceback.print_exc()
            return {**state, "error": f"記事添削エラー: {str(e)}"}

    def format_html_node(state: AgentState) -> AgentState:
        print("--- ステップ5: HTML整形 ---")
        html_title = state.get("initial_article_title") or state.get("topic", "記事")
        html_article_content = state.get("revised_article")
        if not html_article_content:
            html_article_content = state.get("initial_article_content", "")
        if state.get("error") and not html_article_content:
            html_article_content = (
                f"記事のコンテンツ生成に失敗しました。エラー: {state.get('error')}"
            )
        elif not html_article_content:
            html_article_content = "記事が生成されませんでした。"

        # ★ 画像関連のロジックを削除 (image_url, image_alt_text)
        try:
            paragraphs = html_article_content.strip().split("\n\n")
            article_html_paragraphs = "".join(
                [f"<p>{p.strip()}</p>\n" for p in paragraphs if p.strip()]
            )
            # ★ HTMLテンプレートから画像関連部分を削除
            html_output = f"""<!DOCTYPE html><html lang="ja"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>{html_title}</title><style>body {{ font-family: 'Helvetica Neue', Arial, sans-serif; line-height: 1.8; margin: 0; padding: 0; background-color: #f9f9f9; color: #333; }} .container {{ max-width: 800px; margin: 40px auto; background-color: #fff; padding: 30px; border-radius: 12px; box-shadow: 0 6px 20px rgba(0,0,0,0.08); }} h1 {{ font-size: 2.5em; color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 15px; margin-top: 0; margin-bottom: 25px; }} p {{ margin-bottom: 1.5em; font-size: 1.1em; color: #555; }}</style></head><body><div class="container"><h1>{html_title}</h1>{article_html_paragraphs}</div></body></html>"""
            print("HTML整形完了。")
            return {**state, "html_output": html_output, "error": state.get("error")}
        except Exception as e:
            print(f"HTML整形中にエラーが発生しました: {e}")
            traceback.print_exc()
            current_error = state.get("error")
            html_error_msg = f"HTML整形エラー: {str(e)}"
            final_error = (
                f"{current_error}\n{html_error_msg}"
                if current_error
                else html_error_msg
            )
            html_error_output = f"<!DOCTYPE html><html lang='ja'><head><title>エラー</title></head><body><h1>HTML整形中にエラーが発生しました</h1><p><strong>エラー詳細:</strong></p><pre>{final_error}</pre></body></html>"
            return {**state, "html_output": html_error_output, "error": final_error}

    def error_handler_node(state: AgentState) -> AgentState:
        print(f"--- エラー発生 (エラーハンドラノード) ---")
        error_message = state.get("error", "不明なエラー")
        print(f"エラー内容: {error_message}")
        html_error_output = f"""<!DOCTYPE html><html lang="ja"><head><meta charset="UTF-8"><title>処理エラー</title></head><body><h1>記事生成プロセスでエラーが発生しました</h1><p><strong>エラーメッセージ:</strong></p><pre>{error_message}</pre><hr><p><strong>状態情報 (一部):</strong></p><pre>トピック: {state.get("topic")}\n検索クエリ: {state.get("search_query")}\nスクレイプコンテキストの有無: {"あり" if state.get("scraped_context") else "なし"}\n初期記事タイトルの有無: {"あり" if state.get("initial_article_title") else "なし"}</pre></body></html>"""
        return {**state, "html_output": html_error_output}

    # --- グラフ構築 ---
    workflow = StateGraph(AgentState)
    workflow.add_node("generate_search_query", generate_search_query_node)
    workflow.add_node("Google Search", google_search_node)
    workflow.add_node("scrape_and_prepare_context", scrape_and_prepare_context_node)
    workflow.add_node("generate_structured_article", generate_structured_article_node)
    workflow.add_node("revise_article", revise_article_node)
    workflow.add_node("format_html", format_html_node)
    workflow.add_node("error_handler", error_handler_node)

    workflow.set_entry_point("generate_search_query")

    def should_continue_or_handle_error(state: AgentState) -> str:
        if state.get("error"):
            print(
                f"エラー検出: {state.get('error')[:100]}... error_handlerへ遷移します。"
            )
            return "error_handler"
        print("処理継続: 次のノードへ進みます。")
        return "continue"

    workflow.add_conditional_edges(
        "generate_search_query",
        should_continue_or_handle_error,
        {"continue": "Google Search", "error_handler": "error_handler"},
    )
    workflow.add_conditional_edges(
        "Google Search",
        should_continue_or_handle_error,
        {"continue": "scrape_and_prepare_context", "error_handler": "error_handler"},
    )
    workflow.add_conditional_edges(
        "scrape_and_prepare_context",
        should_continue_or_handle_error,
        {"continue": "generate_structured_article", "error_handler": "error_handler"},
    )
    workflow.add_conditional_edges(
        "generate_structured_article",
        should_continue_or_handle_error,
        {"continue": "revise_article", "error_handler": "error_handler"},
    )
    workflow.add_conditional_edges(
        "revise_article",
        should_continue_or_handle_error,
        {"continue": "format_html", "error_handler": "error_handler"},
    )

    workflow.add_edge("format_html", END)
    workflow.add_edge("error_handler", END)

    # --- グラフをコンパイル ---
    try:
        app = workflow.compile()
    except Exception as e_compile:
        error_msg = f"ワークフローグラフのコンパイル中にエラー: {e_compile}"
        print(error_msg)
        traceback.print_exc()
        return {
            "success": False,
            "topic": topic_input,
            "output_file_path": None,
            "error_message": error_msg,
            "final_state_summary": None,
        }

    # --- 初期状態の設定 ---
    # ★ initial_stateから image_url を削除
    initial_state: AgentState = {
        "topic": topic_input,
        "search_query": "",
        "raw_search_results": [],
        "scraped_context": "",
        "generated_article_json": {},
        "initial_article_title": "",
        "initial_article_content": "",
        "revised_article": "",
        "html_output": "",
        "error": None,
        # "image_url": settings.get("default_image_url", "https://placehold.co/600x400/grey/white?text=Image+Not+Generated"), # ★ 削除
    }

    # --- ワークフロー実行 ---
    final_state = None
    try:
        for event_part in app.stream(initial_state, {"recursion_limit": 15}):
            for node_name, current_state_after_node in event_part.items():
                print(f"\n[ノード完了] '{node_name}'")
                print(f"  エラー状態: {current_state_after_node.get('error')}")
                if node_name == "__end__":
                    print("  ワークフロー終了点に到達。")
                final_state = current_state_after_node
    except Exception as e_stream:
        error_msg = f"ワークフロー実行中にエラー: {e_stream}"
        print(error_msg)
        traceback.print_exc()
        # final_stateがNoneの場合も考慮してエラー情報を取得
        final_state_error = None
        if final_state and isinstance(
            final_state, dict
        ):  # final_stateが辞書であることを確認
            final_state_error = str(final_state.get("error"))

        summary_error = final_state_error if final_state_error else error_msg
        summary_dict = {"error": summary_error} if final_state else {"error": error_msg}

        return {
            "success": False,
            "topic": topic_input,
            "output_file_path": None,
            "error_message": error_msg,
            "final_state_summary": summary_dict,
        }

    print("\n--- 全ての処理が完了しました ---")

    # --- 結果の処理とファイル保存 ---
    output_file_path_str = None
    if final_state:
        safe_topic = (
            "".join(c if c.isalnum() or c in [" ", "_"] else "_" for c in topic_input)
            .strip()
            .replace(" ", "_")
        )
        output_filename = f"{safe_topic}_article.html"

        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                print(f"出力ディレクトリを作成しました: {output_dir}")
            except OSError as e_mkdir:
                error_msg = (
                    f"出力ディレクトリの作成に失敗しました ({output_dir}): {e_mkdir}"
                )
                print(error_msg)
                return {
                    "success": False,
                    "topic": topic_input,
                    "output_file_path": None,
                    "error_message": error_msg,
                    "final_state_summary": final_state,
                }

        output_file_path_str = os.path.join(output_dir, output_filename)

        if final_state.get("html_output"):
            try:
                with open(output_file_path_str, "w", encoding="utf-8") as f:
                    f.write(final_state["html_output"])

                if final_state.get("error"):
                    print(
                        f"\n処理中にエラーが発生しましたが、情報を含むHTMLファイル '{output_file_path_str}' を出力しました。"
                    )
                else:
                    print(
                        f"\n最終結果: HTMLファイル '{output_file_path_str}' に出力しました。"
                    )
                print("ブラウザで開いて内容を確認してください。")

                return {
                    "success": not bool(final_state.get("error")),
                    "topic": topic_input,
                    "output_file_path": output_file_path_str,
                    "error_message": final_state.get("error"),
                    "final_state_summary": {
                        k: v
                        for k, v in final_state.items()
                        if k != "raw_search_results"
                    },
                }
            except Exception as e_write:
                error_msg = f"HTMLファイルの書き込み中にエラーが発生しました ({output_file_path_str}): {e_write}"
                print(error_msg)
                traceback.print_exc()
                return {
                    "success": False,
                    "topic": topic_input,
                    "output_file_path": None,
                    "error_message": error_msg,
                    "final_state_summary": final_state,
                }
        else:
            error_msg = "最終結果: HTMLが出力されませんでした。"
            print(error_msg)
            if final_state.get("error"):
                print(f"エラーメッセージ: {final_state.get('error')}")
            return {
                "success": False,
                "topic": topic_input,
                "output_file_path": None,
                "error_message": final_state.get("error", error_msg),
                "final_state_summary": final_state,
            }
    else:
        error_msg = "最終状態が取得できませんでした。"
        print(error_msg)
        return {
            "success": False,
            "topic": topic_input,
            "output_file_path": None,
            "error_message": error_msg,
            "final_state_summary": None,
        }
