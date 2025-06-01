import os
from typing import TypedDict, List, Dict, Any
import traceback

from langchain_google_vertexai import ChatVertexAI


from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException

# Pydanticモデル
from pydantic import BaseModel, Field

from config.env_config import (
    get_env_config,
)

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
    main_title: str
    subtitles: List[str]
    search_query: str
    raw_search_results: List[Dict[str, Any]]
    scraped_context: str
    generated_article_json: Dict[str, Any]
    initial_article_title: str
    initial_article_content: str
    html_output: str
    error: str | None


def generate_article_workflow(
    main_title_input: str,
    subtitles_input: List[str],
) -> Dict[str, Any]:
    """
    指定されたメインタイトルとサブタイトルリストに基づいて記事を生成するワークフローを実行します。
    HTMLコンテンツは返り値の辞書に含まれます。
    """
    print(
        f"\n--- 「{main_title_input}」に関する記事生成を開始します (サブタイトル数: {len(subtitles_input)}) ---"
    )

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

# 重要: 出力形式について
-   出力は、**必ず指定されたJSON形式のデータのみ**としてください。
-   JSONデータの前後に、```json やその他の説明文、余計なテキストを一切含めないでください。
-   **もし何らかの理由で記事全体の生成が困難な場合でも、必ず有効なJSONオブジェクトを返してください。その際は、'title'フィールドに簡潔なエラーメッセージ（例: "コンテンツ生成エラー"）、'block'フィールドに空のリスト（[]）またはエラー詳細を含む単一の文字列のリスト（["詳細なエラー理由"]）を設定してください。絶対に `null` やJSON以外の形式で応答しないでください。**

以下は期待されるJSON構造の例です（内容は適宜変更してください）:
```json
{{
  "title": "生成された記事のメインタイトル",
  "block": [
    "最初のサブタイトルに対応する内容の文章ブロック...",
    "次のサブタイトルに対応する内容の文章ブロック...",
    "（以下、各サブタイトルに対応するブロックが続く）"
  ]
}}
```
上記の例の `title` と `block` の値はあくまでプレースホルダーです。実際の生成内容に置き換えてください。
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
            "html_output": f"<h1>設定エラー</h1><p>{error_msg}</p>",  # エラー時もHTMLを返す
            "error_message": error_msg,
            "final_state_summary": None,
        }

    try:
        # Vertex AIのChatVertexAIモデルを初期化
        # 注意: safety_settings パラメータでコンテンツフィルタリングのレベルを調整できます。
        # LLMからの応答がnullになる場合、これが原因である可能性が高いです。
        # from langchain_google_vertexai.types import HarmCategory, HarmBlockThreshold
        # safety_settings = {
        #     HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
        #     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        #     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        #     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        #     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        # }
        # llm = ChatVertexAI(..., safety_settings=safety_settings)
        # 上記は一例です。適切な設定はユースケースにより異なります。
        # Vertex AI Studioのコンソールでテストし、適切な設定を見つけることを推奨します。
        llm = ChatVertexAI(
            model_name=settings.get(
                "model_name", "gemini-1.0-pro-001"
            ),  # Gemini 1.5 Flash/Pro も検討可能
            temperature=settings.get(
                "temperature", 0.7
            ),  # 設定ファイルから取得できるようにする
            max_output_tokens=settings.get("max_output_tokens", 8192),
            max_retries=settings.get("max_retries", 3),
            stop=None,
            # convert_system_message_to_human=True # Gemini 1.0 Pro はSystemMessageをサポート
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
            "html_output": f"<h1>初期化エラー</h1><p>{error_msg}</p>",
            "error_message": error_msg,
            "final_state_summary": None,
        }

    # --- ノード定義 ---
    def generate_search_query_node(state: AgentState) -> AgentState:
        print("--- ステップ1a: 検索クエリ生成 ---")
        main_title = state["main_title"]
        subtitles = state["subtitles"]
        # サブタイトルがリストであることを確認
        if not isinstance(subtitles, list):
            print(f"警告: subtitlesがリストではありません: {subtitles}")
            subtitles_str = str(subtitles)  # エラー回避のため文字列化
        else:
            subtitles_str = " ".join(subtitles)

        search_query = (
            f"{main_title} {subtitles_str} 詳細 解説 歴史 最新情報 考察 背景 意味 本質"
        )
        print(f"生成された検索クエリ: {search_query}")
        return {**state, "search_query": search_query, "error": None}

    def google_search_node(state: AgentState) -> AgentState:
        if state.get("error"):
            return state
        print("--- ステップ1b: Google検索実行 ---")
        query = state["search_query"]
        num_results = settings.get("num_search_results", 5)
        try:
            print(f"Google検索中 (クエリ: {query}, {num_results}件)...")
            search_results_list = search_wrapper.results(
                query=query, num_results=num_results
            )
            if not search_results_list:  # 検索結果が空の場合
                print("Google検索結果が空でした。")
                return {
                    **state,
                    "raw_search_results": [],
                    "error": None,
                }  # エラーとはしない

            print(f"検索結果 {len(search_results_list)} 件取得完了。")
            return {
                **state,
                "raw_search_results": search_results_list,
                "error": None,
            }
        except Exception as e:
            error_detail = f"Google検索中にエラーが発生しました: {e}"
            print(error_detail)
            # traceback.print_exc() # デバッグ時のみ有効化
            # Google検索エラーの場合、raw_search_resultsを空リストにし、エラーメッセージを設定
            return {
                **state,
                "raw_search_results": [],
                "error": f"Google検索エラー: {str(e)}",
            }

    def scrape_and_prepare_context_node(state: AgentState) -> AgentState:
        if state.get("error"):  # 前のノードでエラーがあればスキップ
            return state
        print("--- ステップ1c: Webスクレイピングとコンテキスト準備 ---")
        search_results_list = state.get(
            "raw_search_results", []
        )  # 前のノードで空リストの可能性あり
        scraped_contents = []
        max_content_length_per_page = settings.get(
            "max_content_length_per_page_scrape", 2000
        )

        if not search_results_list:  # 検索結果が実際にない場合
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
            snippet = result.get(
                "snippet", "スニペットなし"
            )  # スニペットは常に取得できるとは限らない

            # スニペットがない場合は空文字にする
            snippet_text = snippet if snippet else "概要なし"

            if link:
                try:
                    print(f"  読み込み中 ({i+1}/{len(search_results_list)}): {link}")
                    loader = WebBaseLoader(
                        web_path=link,
                        requests_kwargs={
                            "timeout": 20,
                            "headers": {
                                "User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +[http://www.google.com/bot.html](http://www.google.com/bot.html))"
                            },
                        },  # GooglebotのUAを試す
                    )
                    documents = loader.load()
                    if documents:
                        bs_transformer = BeautifulSoupTransformer()
                        # unwanted_tags をより積極的に指定して不要な情報を減らす
                        docs_transformed = bs_transformer.transform_documents(
                            documents,
                            tags_to_extract=[
                                "p",
                                "h1",
                                "h2",
                                "h3",
                                "li",
                                "article",
                                "main",
                                "section",
                                "blockquote",
                                "td",
                                "th",
                            ],  # テーブルセルも追加
                            unwanted_tags=[
                                "script",
                                "style",
                                "nav",
                                "footer",
                                "aside",
                                "form",
                                "header",
                                "figure",
                                "figcaption",
                                "img",
                                "svg",
                                "iframe",
                                "button",
                                "input",
                                "select",
                                "textarea",
                                "label",
                                "link",
                                "meta",
                            ],
                        )
                        page_content_after_transform = " ".join(
                            [
                                doc.page_content.strip()
                                for doc in docs_transformed
                                if doc.page_content and doc.page_content.strip()
                            ]
                        )
                        page_content_after_transform = " ".join(
                            page_content_after_transform.split()
                        )  # Normalize whitespace

                        if page_content_after_transform:
                            shortened_content = page_content_after_transform[
                                :max_content_length_per_page
                            ].strip()
                            if shortened_content:
                                scraped_contents.append(
                                    f"参照元URL: {link}\nタイトル: {title}\n内容の抜粋:\n{shortened_content}"
                                )
                            else:  # 短縮後コンテンツが空
                                scraped_contents.append(
                                    f"参照元URL: {link}\nタイトル: {title}\n概要: {snippet_text}\n(コンテンツ短縮後、空になりました。スニペットを利用します。)"
                                )
                        else:  # transform後コンテンツが空
                            scraped_contents.append(
                                f"参照元URL: {link}\nタイトル: {title}\n概要: {snippet_text}\n(BeautifulSoupTransformer適用後コンテンツが空。スニペットを利用します。)"
                            )
                    else:  # WebBaseLoaderがドキュメントを返さなかった場合
                        scraped_contents.append(
                            f"参照元URL: {link}\nタイトル: {title}\n概要: {snippet_text}\n(WebBaseLoaderがドキュメントを返さず。スニペットを利用します。)"
                        )
                except Exception as e_scrape:
                    print(
                        f"  [詳細エラー] URLからのコンテンツ読み込み/処理エラー: {link} - {e_scrape}"
                    )
                    scraped_contents.append(
                        f"参照元URL: {link}\nタイトル: {title}\n概要: {snippet_text}\n(コンテンツの読み込み/処理中にエラー: {type(e_scrape).__name__}。スニペットを利用します。)"
                    )
            else:  # URLがない場合
                scraped_contents.append(
                    f"タイトル: {title}\n概要: {snippet_text} (URLなし)"
                )

        search_context_str = (
            "\n\n===\n\n".join(scraped_contents)
            if scraped_contents
            else "関連性の高いウェブページのコンテンツは見つかりませんでした。深い考察を行うには情報が不足しています。"
        )
        print(
            f"ウェブページ読み込み完了。最終的なscraped_contextの文字数: {len(search_context_str)}"
        )
        return {**state, "scraped_context": search_context_str, "error": None}

    def generate_structured_article_node(state: AgentState) -> AgentState:
        if state.get("error"):
            return state
        print("--- ステップ2: 構造化記事生成 (哲学的トーン) ---")
        main_title = state["main_title"]
        subtitles_list = state["subtitles"]  # subtitles はリストのはず
        search_context = state["scraped_context"]

        # subtitles_list がリストでない場合のフォールバック
        if not isinstance(subtitles_list, list):
            print(
                f"警告: generate_structured_article_node で subtitles がリストではありません: {subtitles_list}"
            )
            subtitles_for_prompt = (
                f"- 「{str(subtitles_list)}」について考察します。"
                if subtitles_list
                else "- (サブタイトル不明)"
            )
        elif not subtitles_list:  # 空のリストの場合
            subtitles_for_prompt = "- (サブタイトルが指定されていません)"
        else:
            subtitles_for_prompt = "\n".join(
                [f"- 「{s}」について考察します。" for s in subtitles_list]
            )

        max_search_context_len = settings.get(
            "max_search_context_for_llm", 18000
        )  # 少し余裕を持たせる
        if len(search_context) > max_search_context_len:
            print(
                f"検索コンテキストが長すぎるため短縮します。元の長さ: {len(search_context)}, 短縮後: {max_search_context_len}"
            )
            search_context = search_context[:max_search_context_len]

        try:
            system_template = "あなたは洞察力に優れた哲学者であり、同時に言葉を巧みに操るエッセイストです。与えられた情報から本質を抽出し、読者の知的好奇心を刺激し、深い思索へと誘うような、示唆に富んだ文章を構成してください。あなたの文章は、平易でありながらも深遠な問いを投げかけ、読者自身の内省を促す力を持っています。指定されたJSON形式で、各サブタイトルに対応する考察豊かなコンテンツブロックを作成してください。"
            format_instructions = output_parser.get_format_instructions()

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_template),
                    ("user", GENERATE_ARTICLE_PROMPT_TEXT),
                ]
            ).partial(format_instructions=format_instructions)

            chain = prompt | llm | output_parser  # output_parser がここで適用される
            print("LLMによる哲学的記事生成中...")

            input_data = {
                "main_title": main_title,
                "subtitles": subtitles_for_prompt,
                "search_results": search_context,
            }

            print(
                f"LLMへの入力データ (一部): main_title='{main_title}', subtitles_preview='{subtitles_for_prompt[:100]}...', search_results_len={len(search_context)}"
            )

            # LLM呼び出しとPydanticオブジェクトへのパース
            # chain.invoke が LangChain の OutputParserException を投げる可能性がある
            parsed_article_obj = chain.invoke(input_data)

            # parsed_article_objがNoneの場合のチェック (通常OutputParserExceptionで捕捉されるはずだが念のため)
            if parsed_article_obj is None:
                # この状況は、LLMが完全に空か、パース不可能な応答を返し、
                # かつOutputParserがそれをNoneとして処理した場合に発生する可能性がある。
                # 通常、PydanticOutputParserはパース失敗時に例外を投げる。
                print(
                    "致命的エラー: LLMからのパース結果がNoneでした。これは予期されていません。"
                )
                # OutputParserExceptionのllm_outputは、この場合、元のLLMの生の出力（おそらく空や非JSON）であるべき
                raise OutputParserException(
                    "Parsed Article object is None. This typically means the LLM output was empty or unparsable, and the parser handled it as None instead of raising a more specific parsing error.",
                    llm_output="LLM Response led to None after parsing attempt",
                )

            generated_article_json = (
                parsed_article_obj.model_dump()
            )  # Pydanticモデルを辞書に変換
            article_main_title = generated_article_json.get(
                "title", main_title
            )  # フォールバック
            article_blocks = generated_article_json.get("block", [])  # フォールバック

            if (
                not article_main_title and not article_blocks
            ):  # titleもblockも空なら問題あり
                print("LLMがtitleとblockの両方を生成できませんでした。")
                # エラーメッセージを具体的に
                error_msg_content = "LLMが記事のタイトルと本文ブロックの両方を生成できませんでした。プロンプトまたは入力データに問題がある可能性があります。"
                return {
                    **state,
                    "error": f"記事構造生成エラー: {error_msg_content}\nLLM Output (parsed to JSON): {generated_article_json}",
                }

            # サブタイトル数とブロック数の一致チェック（リストの場合のみ）
            if isinstance(subtitles_list, list) and len(article_blocks) != len(
                subtitles_list
            ):
                print(
                    f"警告: 生成されたコンテンツブロック数 ({len(article_blocks)}) がサブタイトル数 ({len(subtitles_list)}) と一致しません。"
                )

            article_combined_content = "\n\n".join(
                article_blocks
            )  # リスト内の文字列を結合
            print(f"哲学的記事生成完了。タイトル: {article_main_title}")

            return {
                **state,
                "generated_article_json": generated_article_json,
                "initial_article_title": article_main_title,
                "initial_article_content": article_combined_content,
                "error": None,
            }
        except OutputParserException as e_parse:
            # e_parse.llm_output にはパース試行前のLLMの生の文字列出力が含まれる
            llm_raw_output = (
                e_parse.llm_output if hasattr(e_parse, "llm_output") else "LLM出力不明"
            )
            if llm_raw_output is None:  # LLMが実際にNoneを返した場合
                llm_raw_output = "LLM returned a null/None response."

            error_detail = f"記事生成中にOutputParserエラー: {str(e_parse)}\nLLM Raw Output:\n{llm_raw_output}"
            print(error_detail)
            return {
                **state,
                "error": f"記事生成パーサーエラー: {str(e_parse)}\nLLM Output: {llm_raw_output}",
            }
        except Exception as e:  # その他の予期せぬエラー
            error_detail = f"記事生成中に予期せぬ汎用エラー: {e}"
            print(error_detail)
            traceback.print_exc()
            return {**state, "error": f"記事生成汎用エラー: {str(e)}"}

    def format_html_node(state: AgentState) -> AgentState:
        print("--- ステップ3: HTML整形 (哲学的記事用) ---")
        html_main_title = state.get("initial_article_title") or state.get(
            "main_title", "考察記事"
        )
        subtitles_list = state.get("subtitles", [])
        # generated_article_json が None や空辞書の可能性も考慮
        generated_json = (
            state.get("generated_article_json")
            if isinstance(state.get("generated_article_json"), dict)
            else {}
        )
        article_content_blocks = (
            generated_json.get("block", [])
            if isinstance(generated_json.get("block"), list)
            else []
        )

        article_html_parts = []
        current_error = state.get("error")

        if current_error:  # 何らかのエラーが発生している場合
            article_html_parts.append(
                f"<h1>思索の途絶</h1><p>記事の構成中に予期せぬ障害が発生しました。</p><p><strong>エラー詳細:</strong></p><pre>{current_error}</pre><hr>"
            )
            # エラーがあっても、部分的なコンテンツがあれば表示を試みる
            if subtitles_list and article_content_blocks:
                article_html_parts.append(
                    "<h2>部分的に生成された可能性のあるコンテンツ:</h2>"
                )
            elif state.get("initial_article_content"):
                article_html_parts.append(
                    "<h2>部分的に生成された可能性のあるコンテンツ（結合済み）:</h2>"
                )

        # subtitles_list と article_content_blocks の整合性を確認して処理
        if (
            subtitles_list
            and article_content_blocks
            and len(subtitles_list) == len(article_content_blocks)
        ):
            for i, subtitle_text in enumerate(subtitles_list):
                article_html_parts.append(f"<h2>{str(subtitle_text).strip()}</h2>\n")
                content_for_this_subtitle = (
                    article_content_blocks[i]
                    if i < len(article_content_blocks)
                    else "コンテンツが生成されませんでした。"
                )
                paragraphs = str(content_for_this_subtitle).strip().split("\n\n")
                for p_content in paragraphs:
                    if p_content.strip():
                        p_content_with_br = p_content.strip().replace("\n", "<br>\n")
                        article_html_parts.append(f"<p>{p_content_with_br}</p>\n")
        elif state.get(
            "initial_article_content"
        ):  # フォールバック (ブロック構造が不正だが結合コンテンツはある場合)
            if (
                not current_error
            ):  # エラーメッセージがまだなければ、構造不一致の警告を追加
                article_html_parts.append(
                    "<p><strong>注意:</strong> 記事の内部構造が期待通りに生成されなかったため、結合された内容を表示します。</p>"
                )
            paragraphs = (
                str(state.get("initial_article_content", "")).strip().split("\n\n")
            )
            for p_content in paragraphs:
                if p_content.strip():
                    p_content_with_br = p_content.strip().replace("\n", "<br>\n")
                    article_html_parts.append(f"<p>{p_content_with_br}</p>\n")
        elif not current_error:  # エラーがなく、コンテンツも全くない場合
            article_html_parts.append(
                "<p>言葉はまだ紡がれていません。記事コンテンツが生成されませんでした。</p>"
            )

        # コンテンツが全くなく、エラーメッセージも表示されていない場合は、汎用メッセージ
        if not article_html_parts:
            article_html_parts.append(
                f"<h1>処理結果</h1><p>記事の生成処理は完了しましたが、表示できるコンテンツがありません。"
                f"{'<br>エラー: ' + current_error if current_error else ''}</p>"
            )

        article_html_body = "".join(article_html_parts)

        html_output_content = f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html_main_title} - 深遠なる考察</title>
<style>
body {{ font-family: 'Merriweather', 'Georgia', serif; line-height: 1.8; margin: 0; padding: 0; background-color: #f0f0f0; color: #2c3e50; }}
.container {{ max-width: 800px; margin: 60px auto; background-color: #ffffff; padding: 50px 60px; border-radius: 2px; box-shadow: 0 8px 25px rgba(0,0,0,0.08); border-top: 5px solid #34495e; }}
h1 {{ font-family: 'Playfair Display', serif; font-size: 2.8em; color: #2c3e50; border-bottom: 2px solid #bdc3c7; padding-bottom: 25px; margin-top: 0; margin-bottom: 40px; font-weight: 700; letter-spacing: 0.8px; text-align: center; }}
h2 {{ font-family: 'Playfair Display', serif; font-size: 2.0em; color: #34495e; margin-top: 50px; margin-bottom: 25px; border-bottom: 1px solid #dfe4ea; padding-bottom: 15px; font-weight: 600; }}
p {{ margin-bottom: 2em; font-size: 1.15em; color: #34495e; text-align: left; orphans: 2; widows: 2; word-wrap: break-word; }}
p br {{ display: block; margin-bottom: 0.5em; content: ""; }}
pre {{ background-color: #f5f5f5; padding: 15px; border: 1px solid #ddd; border-radius: 4px; white-space: pre-wrap; word-wrap: break-word; font-size: 0.9em; }}
blockquote {{ margin: 30px 0; padding: 25px 30px; border-left: 5px solid #3498db; background-color: #f8f9f9; font-style: normal; color: #2c3e50; position: relative; font-size: 1.1em; }}
blockquote::before {{ content: "\\201C"; font-family: 'Georgia', serif; font-size: 4em; color: #3498db; position: absolute; left: 10px; top: -10px; opacity: 0.7; }}
.article-footer {{ margin-top: 50px; padding-top: 25px; border-top: 1px solid #dfe4ea; text-align: center; font-size: 0.95em; color: #7f8c8d; font-style: italic; }}
</style></head>
<body> <div class="container"> <h1>{html_main_title}</h1> {article_html_body} 
<div class="article-footer"> <p>この記事が、あなたの思索の一助となれば幸いです。</p> </div> </div> </body></html>"""
        print("HTML整形完了。")
        return {**state, "html_output": html_output_content, "error": current_error}

    workflow = StateGraph(AgentState)
    workflow.add_node("generate_search_query", generate_search_query_node)
    workflow.add_node("google_search", google_search_node)
    workflow.add_node("scrape_and_prepare_context", scrape_and_prepare_context_node)
    workflow.add_node("generate_structured_article", generate_structured_article_node)
    workflow.add_node("format_html", format_html_node)
    # workflow.add_node("error_handler", error_handler_node) # error_handler は format_html が担う

    workflow.set_entry_point("generate_search_query")

    # 各処理ノードの後にエラーがあれば format_html に遷移し、エラーページを生成
    # エラーがなければ次の処理ノードへ
    def decide_next_step_after_search_query(state: AgentState) -> str:
        return "format_html" if state.get("error") else "google_search"

    def decide_next_step_after_google_search(state: AgentState) -> str:
        return "format_html" if state.get("error") else "scrape_and_prepare_context"

    def decide_next_step_after_scrape(state: AgentState) -> str:
        return "format_html" if state.get("error") else "generate_structured_article"

    def decide_next_step_after_article_gen(state: AgentState) -> str:
        # ここではエラーがあってもなくても format_html に行く
        # format_html がエラーの有無に応じて適切なHTMLを生成する
        return "format_html"

    workflow.add_conditional_edges(
        "generate_search_query", decide_next_step_after_search_query
    )
    workflow.add_conditional_edges(
        "google_search", decide_next_step_after_google_search
    )
    workflow.add_conditional_edges(
        "scrape_and_prepare_context", decide_next_step_after_scrape
    )
    workflow.add_conditional_edges(
        "generate_structured_article", decide_next_step_after_article_gen
    )

    workflow.add_edge("format_html", END)

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
            "html_output": f"<h1>コンパイルエラー</h1><p>{error_msg}</p>",
            "error_message": error_msg,
            "final_state_summary": {"error": error_msg},
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

    final_state_result = None
    try:
        # streamではなくinvokeで最終結果のみ取得する方がシンプルかもしれない
        # for event_part in app.stream(initial_state, {"recursion_limit": 25}):
        #     for node_name, current_state_after_node in event_part.items():
        #         print(f"\n[ノード完了] '{node_name}'")
        #         if node_name == "__end__": print("  ワークフロー終了点に到達。")
        #         final_state_result = current_state_after_node
        final_state_result = app.invoke(initial_state, {"recursion_limit": 15})

    except Exception as e_stream:  # invokeでもエラーは発生しうる
        error_msg = f"ワークフロー実行(invoke)中にエラー: {e_stream}"
        print(error_msg)
        traceback.print_exc()
        # final_state_result が None の可能性もある
        current_html = (
            final_state_result.get("html_output", "")
            if isinstance(final_state_result, dict)
            else ""
        )
        error_html = f"<h1>ワークフロー実行時エラー</h1><p>{error_msg}</p>" + (
            "<hr>" + current_html if current_html else ""
        )
        return {
            "success": False,
            "main_title": main_title_input,
            "subtitles": subtitles_input,
            "html_output": error_html,
            "error_message": error_msg,
            "final_state_summary": {
                "error": error_msg,
                "last_known_state": final_state_result,
            },
        }

    print("\n--- 全ての処理が完了しました ---")
    if final_state_result:
        error_message_from_state = final_state_result.get("error")
        success_status = not bool(error_message_from_state)
        html_content_output = final_state_result.get("html_output", "")

        # サマリーから大きなデータを除外
        final_state_summary_dict = {
            k: (v[:200] + "..." if isinstance(v, str) and len(v) > 200 else v)
            for k, v in final_state_result.items()
            if k
            not in [
                "raw_search_results",
                "scraped_context",
                "html_output",
            ]  # html_outputも除外
        }
        final_state_summary_dict["html_output_preview"] = (
            html_content_output[:200] + "..." if html_content_output else "(HTMLなし)"
        )

        print(
            f"HTML Outputの先頭100文字: {html_content_output[:100] if html_content_output else '（HTMLなし）'}"
        )
        if error_message_from_state:
            print(f"完了時のエラーメッセージ: {error_message_from_state}")

        return {
            "success": success_status,
            "main_title": main_title_input,  # 入力値を返す
            "subtitles": subtitles_input,  # 入力値を返す
            "html_output": html_content_output,
            "error_message": error_message_from_state,  # 最終的なエラーメッセージ
            "final_state_summary": final_state_summary_dict,
        }
    else:
        no_final_state_msg = (
            "最終状態が取得できませんでした (invokeがNoneを返したか、予期せぬエラー)。"
        )
        print(no_final_state_msg)
        return {
            "success": False,
            "main_title": main_title_input,
            "subtitles": subtitles_input,
            "html_output": f"<h1>致命的エラー</h1><p>{no_final_state_msg}</p>",
            "error_message": no_final_state_msg,
            "final_state_summary": {"error": no_final_state_msg},
        }
