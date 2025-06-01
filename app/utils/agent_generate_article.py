import os
from typing import TypedDict, List, Dict, Any
import traceback

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
    topic: str
    search_query: str
    raw_search_results: List[Dict[str, Any]]
    scraped_context: str
    generated_article_json: Dict[str, Any]
    initial_article_title: str
    initial_article_content: str
    # revised_article: str # 添削機能削除のためコメントアウト
    html_output: str  # 生成されたHTMLコンテンツはここに格納される
    error: str | None


def generate_article_workflow(
    topic_input: str,
) -> Dict[str, Any]:
    """
    指定されたトピックに基づいて記事を生成するワークフローを実行します。
    HTMLコンテンツは返り値の辞書に含まれます。

    Args:
        topic_input (str): 記事を生成するトピック。

    Returns:
        Dict[str, Any]: ワークフローの実行結果。以下のキーを含む可能性があります:
            - "success" (bool): 処理が成功したかどうか。
            - "topic" (str): 入力されたトピック。
            - "html_output" (str | None): 生成されたHTMLコンテンツ。エラー時はNoneの場合あり。
            - "error_message" (str | None): エラーが発生した場合のメッセージ。成功時はNone。
            - "final_state_summary" (Dict | None): 最終状態の主要な情報の要約（デバッグ用）。
    """
    print(f"\n--- 「{topic_input}」に関する記事生成を開始します ---")

    settings = get_env_config()
    google_api_key = settings.get("google_api_key")
    google_cse_id = settings.get("google_cse_id")

    if not google_api_key or not google_cse_id:
        error_msg = "エラー: Google APIキーまたはCSE IDが設定ファイルに存在しません。"
        print(error_msg)
        return {
            "success": False,
            "topic": topic_input,
            "html_output": None,
            "error_message": error_msg,
            "final_state_summary": None,
        }

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
            "html_output": None,
            "error_message": error_msg,
            "final_state_summary": None,
        }

    # --- ノード定義 ---
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
        print(f"受け取った検索結果の数: {len(search_results_list)}")
        if search_results_list:
            print(f"最初の検索結果のサンプル: {search_results_list[0]}")

        if not search_results_list:
            print("検索結果が空のため、スクレイピングをスキップします。")
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
                        web_path=link,
                        requests_kwargs={
                            "timeout": 20,
                            "headers": {
                                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                            },
                        },
                    )
                    documents = loader.load()
                    print(
                        f"    URL: {link} - loader.load() 結果のドキュメント数: {len(documents)}"
                    )
                    if documents:
                        print(
                            f"    URL: {link} - WebBaseLoaderで取得した最初のドキュメントの先頭100文字:\n{documents[0].page_content[:100]}"
                        )
                        bs_transformer = BeautifulSoupTransformer()
                        docs_transformed = bs_transformer.transform_documents(
                            documents,
                            tags_to_extract=[
                                "div",
                                "p",
                                "h1",
                                "h2",
                                "h3",
                                "li",
                                "article",
                                "main",
                                "section",
                            ],
                        )
                        print(
                            f"    URL: {link} - bs_transformer.transform_documents() 結果のドキュメント数: {len(docs_transformed)}"
                        )
                        page_content_after_transform = " ".join(
                            [doc.page_content for doc in docs_transformed]
                        )
                        print(
                            f"    URL: {link} - BeautifulSoupTransformer適用後のコンテンツ長: {len(page_content_after_transform)}"
                        )
                        if page_content_after_transform:
                            print(
                                f"    URL: {link} - BeautifulSoupTransformer適用後のコンテンツ先頭100文字:\n{page_content_after_transform[:100]}"
                            )
                        shortened_content = page_content_after_transform[
                            :max_content_length_per_page
                        ].strip()
                        print(
                            f"    URL: {link} - 短縮後のコンテンツ長: {len(shortened_content)}"
                        )
                        if shortened_content:
                            scraped_contents.append(
                                f"参照元URL: {link}\nタイトル: {title}\n内容:\n{shortened_content}"
                            )
                            print(
                                f"    -> コンテンツ取得成功 (先頭{len(shortened_content)}文字)"
                            )
                        else:
                            scraped_contents.append(
                                f"参照元URL: {link}\nタイトル: {title}\n概要: {snippet}\n(主要コンテンツ抽出失敗、または抽出後コンテンツが空)"
                            )
                            print(
                                f"    -> 主要コンテンツ抽出失敗 (抽出後コンテンツが空)、スニペット利用"
                            )
                    else:
                        scraped_contents.append(
                            f"参照元URL: {link}\nタイトル: {title}\n概要: {snippet}\n(コンテンツ取得失敗: WebBaseLoaderがドキュメントを返さず)"
                        )
                        print(
                            f"    -> コンテンツ取得失敗 (WebBaseLoaderがドキュメントを返さず)、スニペット利用"
                        )
                except Exception as e_scrape:
                    print(
                        f"  [詳細エラー] URLからのコンテンツ読み込み/処理エラー: {link}"
                    )
                    traceback.print_exc()
                    scraped_contents.append(
                        f"参照元URL: {link}\nタイトル: {title}\n概要: {snippet}\n(コンテンツの読み込み/処理中にエラーが発生しました: {type(e_scrape).__name__})"
                    )
            else:
                print(f"  URLが提供されていないためスキップ: タイトル '{title}'")
                scraped_contents.append(f"タイトル: {title}\n概要: {snippet} (URLなし)")
        search_context_str = (
            "\n\n===\n\n".join(scraped_contents)
            if scraped_contents
            else "関連性の高いウェブページのコンテンツは見つかりませんでした。"
        )
        print(
            f"ウェブページ読み込み完了。最終的なscraped_contextの先頭200文字: {search_context_str[:200]}"
        )
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

    def format_html_node(state: AgentState) -> AgentState:
        print("--- ステップ3: HTML整形 ---")
        html_title = state.get("initial_article_title") or state.get("topic", "記事")
        html_article_content = state.get("initial_article_content", "")

        if state.get("error") and not html_article_content:
            html_article_content = (
                f"記事のコンテンツ生成に失敗しました。エラー: {state.get('error')}"
            )
        elif not html_article_content:
            html_article_content = "記事が生成されませんでした。"
        try:
            paragraphs = html_article_content.strip().split("\n\n")
            article_html_paragraphs = "".join(
                [f"<p>{p.strip()}</p>\n" for p in paragraphs if p.strip()]
            )
            html_output_content = f"""<!DOCTYPE html>
            <html lang="ja">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{html_title}</title>
                <style>
                    body {{
                        font-family: 'Georgia', 'Times New Roman', serif;
                        line-height: 1.7;
                        margin: 0;
                        padding: 0;
                        background-color: #f4f1ea;
                        color: #3a3a3a;
                    }}
                    .container {{
                        max-width: 750px;
                        margin: 50px auto;
                        background-color: #fffdf7;
                        padding: 40px 50px;
                        border-radius: 4px;
                        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                        border-left: 6px solid #a0522d;
                    }}
                    h1 {{
                        font-family: 'Helvetica Neue', Arial, sans-serif;
                        font-size: 2.4em;
                        color: #4a3b32;
                        border-bottom: 1px solid #dcdcdc;
                        padding-bottom: 20px;
                        margin-top: 0;
                        margin-bottom: 35px;
                        font-weight: bold;
                        letter-spacing: 0.5px;
                    }}
                    p {{
                        margin-bottom: 1.8em;
                        font-size: 1.1em;
                        color: #484848;
                        text-align: justify;
                        orphans: 3;
                        widows: 3;
                    }}
                    blockquote {{
                        margin: 25px 0;
                        padding: 20px 25px 20px 30px;
                        border-left: 4px solid #a0522d;
                        background-color: #f9f6f0;
                        font-style: italic;
                        color: #5a473a;
                        position: relative;
                    }}
                    blockquote::before {{
                        content: "\\201C";
                        font-family: 'Georgia', serif;
                        font-size: 3.5em;
                        color: #a0522d;
                        position: absolute;
                        left: 5px;
                        top: 0px;
                        opacity: 0.8;
                    }}
                    blockquote p {{
                        margin-bottom: 0.5em;
                        font-size: 1em;
                        color: #5a473a;
                    }}
                    blockquote p:last-child {{
                        margin-bottom: 0;
                    }}
                    .article-footer {{
                        margin-top: 40px;
                        padding-top: 20px;
                        border-top: 1px solid #eee;
                        text-align: center;
                        font-size: 0.9em;
                        color: #777;
                        font-style: italic;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>{html_title}</h1>
                    {article_html_paragraphs}
                    </div>
            </body>
            </html>"""
            print("HTML整形完了。")
            return {
                **state,
                "html_output": html_output_content,
                "error": state.get("error"),
            }
        except Exception as e:
            print(f"HTML整形中にエラーが発生しました: {e}")
            traceback.print_exc()
            current_error = state.get("error")
            html_error_msg = f"HTML整形エラー: {str(e)}"
            final_error_msg = (
                f"{current_error}\n{html_error_msg}"
                if current_error
                else html_error_msg
            )
            html_error_output_content = f"<!DOCTYPE html><html lang='ja'><head><title>エラー</title></head><body><h1>HTML整形中にエラーが発生しました</h1><p><strong>エラー詳細:</strong></p><pre>{final_error_msg}</pre></body></html>"
            return {
                **state,
                "html_output": html_error_output_content,
                "error": final_error_msg,
            }

    def error_handler_node(state: AgentState) -> AgentState:
        print(f"--- エラー発生 (エラーハンドラノード) ---")
        error_message = state.get("error", "不明なエラー")
        print(f"エラー内容: {error_message}")
        html_error_output_content = f"""<!DOCTYPE html><html lang="ja"><head><meta charset="UTF-8"><title>処理エラー</title></head><body><h1>記事生成プロセスでエラーが発生しました</h1><p><strong>エラーメッセージ:</strong></p><pre>{error_message}</pre><hr><p><strong>状態情報 (一部):</strong></p><pre>トピック: {state.get("topic")}\n検索クエリ: {state.get("search_query")}\nスクレイプコンテキストの有無: {"あり" if state.get("scraped_context") else "なし"}\n初期記事タイトルの有無: {"あり" if state.get("initial_article_title") else "なし"}</pre></body></html>"""
        return {
            **state,
            "html_output": html_error_output_content,
        }

    # --- グラフ構築 ---
    workflow = StateGraph(AgentState)
    workflow.add_node("generate_search_query", generate_search_query_node)
    workflow.add_node("google_search", google_search_node)
    workflow.add_node("scrape_and_prepare_context", scrape_and_prepare_context_node)
    workflow.add_node("generate_structured_article", generate_structured_article_node)
    workflow.add_node("format_html", format_html_node)
    workflow.add_node("error_handler", error_handler_node)

    workflow.set_entry_point("generate_search_query")

    def should_continue_or_handle_error(state: AgentState) -> str:
        if state.get("error"):
            print(
                f"エラー検出: {state.get('error')[:100]}... error_handlerへ遷移します。"  # type: ignore
            )
            return "error_handler"
        print("処理継続: 次のノードへ進みます。")
        return "continue"

    workflow.add_conditional_edges(
        "generate_search_query",
        should_continue_or_handle_error,
        {"continue": "google_search", "error_handler": "error_handler"},
    )
    workflow.add_conditional_edges(
        "google_search",
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
            "html_output": None,
            "error_message": error_msg,
            "final_state_summary": None,
        }

    # --- 初期状態の設定 ---
    initial_state: AgentState = {
        "topic": topic_input,
        "search_query": "",
        "raw_search_results": [],
        "scraped_context": "",
        "generated_article_json": {},
        "initial_article_title": "",
        "initial_article_content": "",
        # "revised_article": "", # 添削機能削除のためコメントアウト
        "html_output": "",
        "error": None,
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
        current_html_output = (
            final_state.get("html_output") if isinstance(final_state, dict) else None
        )
        current_error_message = (
            final_state.get("error") if isinstance(final_state, dict) else None
        )
        final_error_message = (
            current_error_message if current_error_message else error_msg
        )
        summary_dict = {
            "error": final_error_message,
            "topic": topic_input,
            "html_output": current_html_output,
        }
        return {
            "success": False,
            "topic": topic_input,
            "html_output": current_html_output,
            "error_message": final_error_message,
            "final_state_summary": summary_dict,
        }

    print("\n--- 全ての処理が完了しました ---")

    if final_state:
        error_message_from_state = final_state.get("error")
        success_status = not bool(error_message_from_state)
        html_content_output = final_state.get("html_output", "")

        final_state_summary_dict = {
            k: v
            for k, v in final_state.items()
            if k != "raw_search_results"  # 生の検索結果は大きすぎるので除外
        }
        if "html_output" not in final_state_summary_dict:
            final_state_summary_dict["html_output"] = html_content_output

        print(
            f"HTML Outputの先頭100文字: {html_content_output[:100] if html_content_output else '（HTMLなし）'}"
        )

        return {
            "success": success_status,
            "topic": topic_input,
            "html_output": html_content_output,
            "error_message": error_message_from_state,
            "final_state_summary": final_state_summary_dict,
        }
    else:
        error_msg_no_final_state = (
            "最終状態が取得できませんでした (ワークフロー実行で予期せぬ問題)。"
        )
        print(error_msg_no_final_state)
        return {
            "success": False,
            "topic": topic_input,
            "html_output": None,
            "error_message": error_msg_no_final_state,
            "final_state_summary": None,
        }
