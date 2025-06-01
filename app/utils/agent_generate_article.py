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

from config.env_config import get_env_config

from langchain_google_community.search import GoogleSearchAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_transformers import BeautifulSoupTransformer

# LangGraph
from langgraph.graph import StateGraph, END


# --- Pydanticモデル定義 (記事構造) ---
class Article(BaseModel):
    title: str = Field(description="記事のタイトル (メインタイトル)")
    block: List[str] = Field(
        description="記事の各ブロックの本文リスト。各要素がサブタイトルに対応するコンテンツとなることを期待。"
    )


# --- 状態の定義 ---
class AgentState(TypedDict):
    main_title: str  # メインタイトル
    subtitles: List[str]  # サブタイトルのリスト
    search_query: str
    raw_search_results: List[Dict[str, Any]]
    scraped_context: str
    generated_article_json: Dict[str, Any]  # LLMが生成した構造化記事データ
    initial_article_title: str  # LLMによって生成された、または入力されたメインタイトル
    initial_article_content: str  # 全サブタイトルコンテンツを結合したもの
    html_output: str
    error: str | None


def generate_article_workflow(
    main_title_input: str,
    subtitles_input: List[str],
) -> Dict[str, Any]:
    """
    指定されたメインタイトルとサブタイトルリストに基づいて記事を生成するワークフローを実行します。
    HTMLコンテンツは返り値の辞書に含まれます。

    Args:
        main_title_input (str): 記事のメインタイトル。
        subtitles_input (List[str]): 記事のサブタイトルのリスト。

    Returns:
        Dict[str, Any]: ワークフローの実行結果。以下のキーを含む可能性があります:
            - "success" (bool): 処理が成功したかどうか。
            - "main_title" (str): 入力されたメインタイトル。
            - "subtitles" (List[str]): 入力されたサブタイトルリスト。
            - "html_output" (str | None): 生成されたHTMLコンテンツ。エラー時はNoneの場合あり。
            - "error_message" (str | None): エラーが発生した場合のメッセージ。成功時はNone。
            - "final_state_summary" (Dict | None): 最終状態の主要な情報の要約（デバッグ用）。
    """
    print(
        f"\n--- 「{main_title_input}」に関する記事生成を開始します (サブタイトル数: {len(subtitles_input)}) ---"
    )

    # プロンプトをコード内に直接定義
    GENERATE_ARTICLE_PROMPT_TEXT = """
以下の検索結果、メインタイトル、およびサブタイトルリストに基づいて、高品質なブログ記事を作成してください。
あなたの出力は、JSON形式で、指示された構造に従う必要があります。
{format_instructions}

## 検索結果
{search_results}

## メインタイトル
{main_title}

## サブタイトルリスト (このリストの各項目に対してコンテンツブロックを作成してください)
{subtitles}

### 記事作成の詳細指示
- 読者層: 若い層、知的好奇心が旺盛な層
- 文章トーン: 親しみやすさを保ちつつ、洞察に満ちた哲学的思索を促すような、示唆に富むスタイル。単なる情報提供に留まらず、読者が物事の本質について深く考えるきっかけを与えるような、余韻の残る文章を心がけてください。
- 記事の目的: 「{main_title}」に関する情報を、表層的な解説ではなく、多角的な視点から深く掘り下げて分かりやすく伝え、読者の知的好奇心を刺激し、内省を促す。
- 文章の長さ: 各ブロック (各サブタイトルに対応) 300〜400文字程度。言葉を慎重に選び、簡潔かつ深みのある表現を目指してください。

---

## 作成プロセス
1.  **記事全体のタイトル決定**:
    * メインタイトル「{main_title}」を参考に、記事全体のテーマ性を捉え、読者の興味を惹きつけるような、示唆的かつ魅力的なタイトルを決定してください。これはPydanticモデルの 'title' フィールドに対応します。

2.  **各サブタイトルのコンテンツ作成**:
    * 上記の「サブタイトルリスト」に含まれる各サブタイトルについて、順番にコンテンツブロックを作成してください。これらはPydanticモデルの 'block' リストの各要素に対応します。
    * 各サブタイトルのコンテンツを作成する際は、以下の点を特に重視してください:
        * **深い洞察**: 検索結果や関連情報を単に要約するのではなく、それらの情報から本質を見抜き、独自の哲学的考察や解釈を加える。表面的な事象の奥にある意味や関連性を示唆する。
        * **問いかける姿勢**: 読者に対して問いを投げかけ、自ら考えることを促すような記述を適度に含める。断定的な表現よりも、多様な解釈の可能性を示唆するようなニュアンスを大切にする。
        * **言葉の選択**: 平易で理解しやすい言葉を選びつつも、表現には深みと詩的な響きを持たせる。比喩や隠喩を効果的に用い、読者の想像力をかき立てる。
        * **構成の妙**: 各ブロック内で、導入、展開、そして思索を促すような結論へと、論理的かつ魅力的に物語を紡ぐ。
        * 信頼性のある情報源からの情報を適切に含める。
        * SEOを意識し、関連キーワードを自然な形で盛り込む。
        * 読みやすさを最優先し、短い段落を適宜使用する。しかし、各段落は意味のあるまとまりを持つようにする。

---

# 注意
- 出力は必ず上記のフォーマット指示に厳密に従ったJSONデータのみを出力してください。
- ```json やその他の余計なテキストをJSONデータ本体の前後に含めないでください。
"""

    settings = get_env_config()
    google_api_key = settings.get("google_api_key")
    google_cse_id = settings.get("google_cse_id")

    if not google_api_key or not google_cse_id:
        error_msg = "エラー: Google APIキーまたはCSE IDが設定ファイルに存在しません。"
        print(error_msg)
        return {
            "success": False,
            "main_title": main_title_input,
            "subtitles": subtitles_input,
            "html_output": None,
            "error_message": error_msg,
            "final_state_summary": None,
        }

    try:
        llm = ChatVertexAI(
            model_name=settings.get("model_name", "gemini-1.0-pro-001"),
            temperature=0.7,  # 哲学的な文章生成のために少し温度を上げることを検討 (0.5-0.8程度)
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
            "main_title": main_title_input,
            "subtitles": subtitles_input,
            "html_output": None,
            "error_message": error_msg,
            "final_state_summary": None,
        }

    # --- ノード定義 ---
    def generate_search_query_node(state: AgentState) -> AgentState:
        print("--- ステップ1a: 検索クエリ生成 ---")
        main_title = state["main_title"]
        subtitles = state["subtitles"]
        query_parts = [main_title] + subtitles
        search_query = f"{main_title} {' '.join(subtitles)} 詳細 解説 歴史 最新情報 考察 背景"  # 哲学的な内容を示唆するキーワード追加
        print(f"生成された検索クエリ: {search_query}")
        return {**state, "search_query": search_query, "error": None}

    def google_search_node(state: AgentState) -> AgentState:
        if state.get("error"):
            return state
        print("--- ステップ1b: Google検索実行 ---")
        query = state["search_query"]
        num_results = settings.get(
            "num_search_results", 5
        )  # 考察の幅を広げるため少し多めに情報を取得することも検討
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
            traceback.print_exc()
            return {**state, "error": f"Google検索エラー: {str(e)}"}

    def scrape_and_prepare_context_node(state: AgentState) -> AgentState:
        if state.get("error"):
            return state
        print("--- ステップ1c: Webスクレイピングとコンテキスト準備 ---")
        search_results_list = state["raw_search_results"]
        scraped_contents = []
        max_content_length_per_page = settings.get(
            "max_content_length_per_page_scrape",
            2000,  # 考察の材料を増やすため少し長めに取得
        )
        print(f"受け取った検索結果の数: {len(search_results_list)}")
        if search_results_list:
            print(f"最初の検索結果のサンプル: {search_results_list[0]}")

        if not search_results_list:
            print("検索結果が空のため、スクレイピングをスキップします。")
            return {
                **state,
                "scraped_context": "関連情報は見つかりませんでした。深い考察を行うには情報が不足しています。",
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
                                "span",
                                "blockquote",  # 引用も抽出対象に
                            ],
                            unwanted_tags=[
                                "script",
                                "style",
                                "nav",
                                "footer",
                                "aside",
                                "form",
                                "header",
                            ],  # 不要なタグをより具体的に
                        )
                        print(
                            f"    URL: {link} - bs_transformer.transform_documents() 結果のドキュメント数: {len(docs_transformed)}"
                        )
                        page_content_after_transform = " ".join(
                            [
                                doc.page_content.strip()
                                for doc in docs_transformed
                                if doc.page_content.strip()
                            ]
                        )
                        page_content_after_transform = " ".join(
                            page_content_after_transform.split()
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
                                    f"参照元URL: {link}\nタイトル: {title}\n内容の抜粋:\n{shortened_content}"
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
                                f"参照元URL: {link}\nタイトル: {title}\n概要: {snippet}\n(BeautifulSoupTransformer適用後コンテンツが空)"
                            )
                            print(
                                f"    -> BeautifulSoupTransformer適用後コンテンツが空、スニペット利用"
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
            else "関連性の高いウェブページのコンテンツは見つかりませんでした。深い考察を行うには情報が不足しています。"
        )
        print(
            f"ウェブページ読み込み完了。最終的なscraped_contextの先頭200文字: {search_context_str[:200]}"
        )
        return {**state, "scraped_context": search_context_str, "error": None}

    def generate_structured_article_node(state: AgentState) -> AgentState:
        if state.get("error"):
            return state
        print("--- ステップ2: 構造化記事生成 (哲学的トーン) ---")
        main_title = state["main_title"]
        subtitles = state["subtitles"]
        search_context = state["scraped_context"]

        try:
            # システムプロンプトも哲学的なトーンを意識させる
            system_template = "あなたは洞察力に優れた哲学者であり、同時に言葉を巧みに操るエッセイストです。与えられた情報から本質を抽出し、読者の知的好奇心を刺激し、深い思索へと誘うような、示唆に富んだ文章を構成してください。あなたの文章は、平易でありながらも深遠な問いを投げかけ、読者自身の内省を促す力を持っています。指定されたJSON形式で、各サブタイトルに対応する考察豊かなコンテンツブロックを作成してください。"
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_template),
                    ("user", GENERATE_ARTICLE_PROMPT_TEXT),
                ]
            ).partial(format_instructions=output_parser.get_format_instructions())

            chain = prompt | llm | output_parser
            print("LLMによる哲学的記事生成中...")

            # サブタイトルリストをLLMに分かりやすいように整形して渡すことも検討
            # 例: formatted_subtitles = "\n".join([f"- {s}" for s in subtitles])
            input_data = {
                "main_title": main_title,
                "subtitles": "\n".join(
                    [f"- 「{s}」について" for s in subtitles]
                ),  # LLMがリストとして解釈しやすいように
                "search_results": search_context,
            }

            parsed_article_obj: Article = chain.invoke(input_data)
            generated_article_json = parsed_article_obj.model_dump()

            article_main_title = generated_article_json.get("title", main_title)
            article_blocks = generated_article_json.get("block", [])

            if len(article_blocks) != len(subtitles):
                print(
                    f"警告: 生成されたコンテンツブロック数 ({len(article_blocks)}) がサブタイトル数 ({len(subtitles)}) と一致しません。不足分は空のブロックとして扱われる可能性があります。"
                )
                # 不足している場合、エラーにするか、空のブロックで補完するかなどの対応が必要
                # ここでは、HTML生成側でブロック数がサブタイトル数と一致しない場合の処理を想定

            article_combined_content = "\n\n".join(article_blocks)

            print(f"哲学的記事生成完了。タイトル: {article_main_title}")
            if article_blocks:
                print(f"最初のコンテンツブロックの先頭50文字: {article_blocks[0][:50]}")

            return {
                **state,
                "generated_article_json": generated_article_json,
                "initial_article_title": article_main_title,
                "initial_article_content": article_combined_content,
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
            traceback.print_exc()
            return {
                **state,
                "error": f"記事生成パーサーエラー: {str(e_parse)}\nLLM Output: {llm_output_str}",
            }
        except Exception as e:
            print(f"記事生成中に予期せぬエラーが発生しました: {e}")
            traceback.print_exc()
            return {**state, "error": f"記事生成エラー: {str(e)}"}

    def format_html_node(state: AgentState) -> AgentState:
        print("--- ステップ3: HTML整形 (哲学的記事用) ---")

        html_main_title = state.get("initial_article_title") or state.get(
            "main_title", "考察記事"
        )
        subtitles_list = state.get("subtitles", [])
        generated_json = state.get("generated_article_json", {})
        article_content_blocks = generated_json.get("block", [])

        article_html_parts = []

        # エラー発生時、かつ整形できるコンテンツが期待通りでない場合の処理を強化
        if state.get("error") and not (
            subtitles_list
            and article_content_blocks
            and len(subtitles_list) == len(article_content_blocks)
        ):
            error_message = state.get("error", "不明なエラー")
            article_html_parts.append(
                f"<h1>思索の途絶</h1><p>記事の構成中に予期せぬ障害が発生しました。</p><p>詳細: {error_message}</p>"
            )
        elif (
            subtitles_list
            and article_content_blocks
            and len(subtitles_list) == len(article_content_blocks)
        ):
            for i, subtitle_text in enumerate(subtitles_list):
                article_html_parts.append(f"<h2>{subtitle_text.strip()}</h2>\n")
                content_for_this_subtitle = article_content_blocks[i]
                # 哲学的な文章は改行が意味を持つ場合があるので、\n\nだけでなく\nも<br>に変換することを検討
                # ここではシンプルに段落分割
                paragraphs = content_for_this_subtitle.strip().split("\n\n")
                for p_content in paragraphs:
                    if p_content.strip():
                        # 段落内の改行を <br> に変換して、詩的な表現の改行を保持
                        p_content_with_br = p_content.strip().replace("\n", "<br>\n")
                        article_html_parts.append(f"<p>{p_content_with_br}</p>\n")
        elif state.get("initial_article_content"):
            print(
                "HTML整形: サブタイトルとブロック構造が期待通りでないため、結合コンテンツをフラットに表示します。"
            )
            if state.get("error"):
                error_message = state.get("error", "不明なエラー")
                article_html_parts.append(
                    f"<p><strong>警告:</strong> 部分的なエラーが発生した可能性があります。エラー詳細: {error_message}</p>"
                )
            paragraphs = state.get("initial_article_content", "").strip().split("\n\n")
            for p_content in paragraphs:
                if p_content.strip():
                    p_content_with_br = p_content.strip().replace("\n", "<br>\n")
                    article_html_parts.append(f"<p>{p_content_with_br}</p>\n")
        else:  # コンテンツが全くない場合
            article_html_parts.append("<p>言葉はまだ紡がれていません。</p>")
            if state.get("error"):
                error_message = state.get("error", "不明なエラー")
                article_html_parts.insert(
                    0,
                    f"<h1>思索の途絶</h1><p>記事の構成中に予期せぬ障害が発生しました。</p><p>詳細: {error_message}</p>",
                )

        article_html_body = "".join(article_html_parts)

        # CSSも哲学的雰囲気に合わせて調整
        html_output_content = f"""<!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{html_main_title} - 深遠なる考察</title>
            <style>
                body {{
                    font-family: 'Merriweather', 'Georgia', serif; /* 読みやすく格調高いフォント */
                    line-height: 1.8; /* 行間を広めに */
                    margin: 0;
                    padding: 0;
                    background-color: #f0f0f0; /* やや落ち着いた背景色 */
                    color: #2c3e50; /* 深みのあるテキスト色 */
                }}
                .container {{
                    max-width: 800px; /* 少し広めのコンテナ */
                    margin: 60px auto;
                    background-color: #ffffff; 
                    padding: 50px 60px;
                    border-radius: 2px; /* シャープな印象 */
                    box-shadow: 0 8px 25px rgba(0,0,0,0.08);
                    border-top: 5px solid #34495e; /* アクセントカラー */
                }}
                h1 {{
                    font-family: 'Playfair Display', serif; /* エレガントな見出しフォント */
                    font-size: 2.8em;
                    color: #2c3e50; 
                    border-bottom: 2px solid #bdc3c7; /* 細めの区切り線 */
                    padding-bottom: 25px;
                    margin-top: 0;
                    margin-bottom: 40px;
                    font-weight: 700; /* 太字 */
                    letter-spacing: 0.8px;
                    text-align: center;
                }}
                h2 {{
                    font-family: 'Playfair Display', serif;
                    font-size: 2.0em;
                    color: #34495e; 
                    margin-top: 50px; 
                    margin-bottom: 25px; 
                    border-bottom: 1px solid #dfe4ea; 
                    padding-bottom: 15px;
                    font-weight: 600; 
                }}
                p {{
                    margin-bottom: 2em;
                    font-size: 1.15em; /* 少し大きめのフォント */
                    color: #34495e; 
                    text-align: left; /* 左揃えで落ち着いた印象 */
                    orphans: 2;
                    widows: 2;
                }}
                p br {{ /* 段落内改行のスペース調整 */
                    display: block;
                    margin-bottom: 0.5em; 
                    content: "";
                }}
                blockquote {{
                    margin: 30px 0;
                    padding: 25px 30px;
                    border-left: 5px solid #3498db; /* 引用のアクセントカラー */
                    background-color: #f8f9f9; 
                    font-style: normal; /* イタリック解除、内容で表現 */
                    color: #2c3e50;
                    position: relative;
                    font-size: 1.1em;
                }}
                blockquote::before {{
                    content: "\\201C"; 
                    font-family: 'Georgia', serif;
                    font-size: 4em;
                    color: #3498db;
                    position: absolute;
                    left: 10px;
                    top: -10px; /* 位置調整 */
                    opacity: 0.7;
                }}
                blockquote p {{
                    margin-bottom: 0.8em;
                    font-size: 1em;
                    color: #2c3e50;
                }}
                blockquote p:last-child {{
                    margin-bottom: 0;
                }}
                .article-footer {{
                    margin-top: 50px;
                    padding-top: 25px;
                    border-top: 1px solid #dfe4ea;
                    text-align: center;
                    font-size: 0.95em;
                    color: #7f8c8d; /* フッターのテキスト色 */
                    font-style: italic;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{html_main_title}</h1>
                {article_html_body}
                <div class="article-footer">
                    <p>この記事が、あなたの思索の一助となれば幸いです。</p>
                </div>
            </div>
        </body>
        </html>"""
        print("HTML整形完了。")
        return {
            **state,
            "html_output": html_output_content,
            "error": state.get("error"),
        }

    def error_handler_node(state: AgentState) -> AgentState:
        print(f"--- エラー発生 (エラーハンドラノード) ---")
        error_message = state.get("error", "不明なエラー")
        print(f"エラー内容: {error_message}")

        subtitles_str = (
            ", ".join(state.get("subtitles", [])) if state.get("subtitles") else "なし"
        )

        html_error_output_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>処理エラー</title>
    <style> body {{ font-family: sans-serif; margin: 20px; background-color: #f0f0f0; color: #333; }} 
            .container {{ max-width: 800px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #d32f2f; }} pre {{ background-color: #eee; padding: 15px; border: 1px solid #ddd; border-radius: 4px; white-space: pre-wrap; word-wrap: break-word;}} </style>
</head>
<body>
    <div class="container">
        <h1>記事生成プロセスでエラーが発生しました</h1>
        <p><strong>エラーメッセージ:</strong></p>
        <pre>{error_message}</pre>
        <hr>
        <p><strong>状態情報 (一部):</strong></p>
        <pre>
メインタイトル: {state.get("main_title", "N/A")}
サブタイトル: {subtitles_str}
検索クエリ: {state.get("search_query", "N/A")}
スクレイプコンテキストの有無: {"あり" if state.get("scraped_context") else "なし"}
生成記事JSONの有無: {"あり" if state.get("generated_article_json") else "なし"}
初期記事タイトルの有無: {"あり" if state.get("initial_article_title") else "なし"}
        </pre>
    </div>
</body>
</html>"""
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
            error_preview = str(state.get("error", ""))[:100]
            print(f"エラー検出: {error_preview}... error_handlerへ遷移します。")
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
    # generate_structured_article でエラーが発生した場合、format_html にエラー情報を渡し、
    # format_html 側でエラーに応じたHTML（エラーメッセージを含むHTML）を生成するようにする。
    workflow.add_conditional_edges(
        "generate_structured_article",
        should_continue_or_handle_error,  # この判定は state['error'] を見る
        {
            "continue": "format_html",  # エラーがなければ通常通り format_html へ
            "error_handler": "format_html",  # エラーがあっても format_html へ遷移し、エラー情報を渡す
        },
    )
    workflow.add_edge("format_html", END)  # format_html は常に終点へ
    workflow.add_edge(
        "error_handler", END
    )  # error_handler も常に終点へ (ただし、通常は format_html がエラー処理を行う)

    try:
        app = workflow.compile()
    except Exception as e_compile:
        error_msg = f"ワークフローグラフのコンパイル中にエラー: {e_compile}"
        print(error_msg)
        traceback.print_exc()
        return {
            "success": False,
            "main_title": main_title_input,
            "subtitles": subtitles_input,
            "html_output": None,
            "error_message": error_msg,
            "final_state_summary": None,
        }

    initial_state: AgentState = {
        "main_title": main_title_input,
        "subtitles": subtitles_input,
        "search_query": "",
        "raw_search_results": [],
        "scraped_context": "",
        "generated_article_json": {},
        "initial_article_title": "",
        "initial_article_content": "",
        "html_output": "",
        "error": None,
    }

    final_state = None
    try:
        for event_part in app.stream(initial_state, {"recursion_limit": 25}):
            for node_name, current_state_after_node in event_part.items():
                print(f"\n[ノード完了] '{node_name}'")
                # print(f"  エラー状態: {current_state_after_node.get('error')}") # デバッグ用
                if node_name == "__end__":
                    print("  ワークフロー終了点に到達。")
                final_state = current_state_after_node
    except Exception as e_stream:
        error_msg = f"ワークフロー実行中にエラー: {e_stream}"
        print(error_msg)
        traceback.print_exc()

        current_html_output = (
            final_state.get("html_output", "")
            if isinstance(final_state, dict)
            else f"<h1>実行時エラー</h1><p>{error_msg}</p>"
        )
        current_error_message = (
            final_state.get("error") if isinstance(final_state, dict) else None
        )

        final_error_message_for_return = (
            current_error_message if current_error_message else error_msg
        )

        summary_dict = {
            "error": final_error_message_for_return,
            "main_title": main_title_input,
            "subtitles": subtitles_input,
            "html_output_on_error": current_html_output,
        }
        return {
            "success": False,
            "main_title": main_title_input,
            "subtitles": subtitles_input,
            "html_output": current_html_output,
            "error_message": final_error_message_for_return,
            "final_state_summary": summary_dict,
        }

    print("\n--- 全ての処理が完了しました ---")

    if final_state:
        error_message_from_state = final_state.get("error")
        success_status = not bool(error_message_from_state)
        html_content_output = final_state.get("html_output", "")

        final_state_summary_dict = {
            k: (v[:200] + "..." if isinstance(v, str) and len(v) > 200 else v)
            for k, v in final_state.items()
            if k not in ["raw_search_results", "scraped_context"]
        }
        if "html_output" not in final_state_summary_dict:
            final_state_summary_dict["html_output_preview"] = (
                html_content_output[:200] + "..." if html_content_output else ""
            )

        print(
            f"HTML Outputの先頭100文字: {html_content_output[:100] if html_content_output else '（HTMLなし）'}"
        )
        if error_message_from_state:
            print(f"完了時のエラーメッセージ: {error_message_from_state}")

        return {
            "success": success_status,
            "main_title": main_title_input,
            "subtitles": subtitles_input,
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
            "main_title": main_title_input,
            "subtitles": subtitles_input,
            "html_output": f"<h1>致命的なエラー</h1><p>{error_msg_no_final_state}</p>",
            "error_message": error_msg_no_final_state,
            "final_state_summary": None,
        }
