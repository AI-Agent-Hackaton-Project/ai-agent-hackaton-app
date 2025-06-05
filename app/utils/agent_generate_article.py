# article_workflow.py
import os
import traceback
from typing import TypedDict, List, Dict, Any

from langchain_google_vertexai import ChatVertexAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field

from config.env_config import get_env_config

from langchain_google_community.search import GoogleSearchAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_transformers import BeautifulSoupTransformer

from langgraph.graph import StateGraph, END

from utils.generate_four_images import generate_four_images


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
    initial_article_title: str  # LLMが生成した記事のタイトル
    initial_article_content: str  # LLMが生成した記事の結合されたブロックコンテンツ
    prefecture_image_path: str | None  # 生成された都道府県画像のパス
    html_output: str
    error: str | None


def generate_article_workflow(
    main_title_input: str,
    subtitles_input: List[str],
    attempt_prefecture_image: bool = True,  # Flag to control image generation attempt
) -> Dict[str, Any]:
    """
    指定されたメインタイトルとサブタイトルリストに基づいて記事を生成するワークフローを実行します。
    HTMLコンテンツは返り値の辞書に含まれます。
    `attempt_prefecture_image` がTrueの場合、`main_title_input`を県名として画像生成を試みます。
    """
    print(
        f"\n--- 「{main_title_input}」に関する記事生成を開始します (サブタイトル数: {len(subtitles_input)}) ---"
    )
    if attempt_prefecture_image:
        print(f"--- 都道府県画像の生成を試みます (対象: {main_title_input}) ---")

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
            "html_output": f"<h1>設定エラー</h1><p>{error_msg}</p>",
            "error_message": error_msg,
            "final_state_summary": None,
        }

    try:
        llm = ChatVertexAI(
            model_name=settings.get("model_name", "gemini-1.5-flash-001"),
            project=settings.get("gcp_project_id"),
            location=settings.get("gcp_location"),
            temperature=settings.get("temperature", 0.7),
            max_output_tokens=settings.get("max_output_tokens", 8192),
            max_retries=settings.get("max_retries", 3),
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
            "html_output": f"<h1>初期化エラー</h1><p>{error_msg}</p>",
            "error_message": error_msg,
            "final_state_summary": None,
        }

    # --- ノード定義 ---
    def generate_prefecture_image_node(state: AgentState) -> AgentState:
        if not attempt_prefecture_image:
            print("--- 都道府県画像生成スキップ (フラグOFF) ---")
            return {**state, "prefecture_image_path": None, "error": state.get("error")}

        # Preserve existing errors, but attempt image generation if flag is true
        current_error_from_previous_steps = state.get(
            "error"
        )  # Error from steps *before* image gen (should be None if this is first)

        print("--- ステップ0: 都道府県画像生成 (該当する場合) ---")
        prefecture_name_for_image = state[
            "main_title"
        ]  # Assume main_title is the prefecture
        try:
            print(f"画像生成関数を呼び出します (対象: {prefecture_name_for_image})...")
            image_path = generate_four_images(
                prefecture_name_for_image
            )
            if image_path:
                print(f"都道府県画像が生成され、一時保存されました: {image_path}")
                return {
                    **state,
                    "prefecture_image_path": image_path,
                    "error": current_error_from_previous_steps,
                }  # Preserve previous error
            else:
                print(
                    f"都道府県画像の生成に失敗しましたが、パスは返されませんでした (対象: {prefecture_name_for_image})。"
                )
                # Do not set/override state["error"] here to allow article generation to continue
                # This specific failure (no path) is logged but not treated as a blocking error for text.
                return {
                    **state,
                    "prefecture_image_path": None,
                    "error": current_error_from_previous_steps,
                }
        except Exception as e_img:
            error_detail_img = f"都道府県画像の生成中にエラーが発生しました (対象: {prefecture_name_for_image}): {e_img}"
            print(error_detail_img)
            # traceback.print_exc() # Uncomment for detailed debugging
            # Store error and append to existing errors if any
            updated_error = (
                f"{current_error_from_previous_steps}\n{error_detail_img}"
                if current_error_from_previous_steps
                else error_detail_img
            )
            return {**state, "prefecture_image_path": None, "error": updated_error}

    def generate_search_query_node(state: AgentState) -> AgentState:
        current_error = state.get(
            "error"
        )  # Preserve errors from image gen or previous steps
        print("--- ステップ1a: 検索クエリ生成 ---")
        main_title = state["main_title"]
        subtitles = state["subtitles"]
        if not isinstance(subtitles, list):
            print(f"警告: subtitlesがリストではありません: {subtitles}")
            subtitles_str = str(subtitles)
        else:
            subtitles_str = " ".join(subtitles)

        search_query = (
            f"{main_title} {subtitles_str} 詳細 解説 歴史 最新情報 考察 背景 意味 本質"
        )
        print(f"生成された検索クエリ: {search_query}")
        return {**state, "search_query": search_query, "error": current_error}

    def google_search_node(state: AgentState) -> AgentState:
        current_error = state.get("error")
        print("--- ステップ1b: Google検索実行 ---")
        query = state["search_query"]
        try:
            print(f"Google検索中 (クエリ: {query})...")
            search_results_list = search_wrapper.results(
                query=query,
                num_results=settings.get("search_num_results", 5),
            )

            if not search_results_list:
                print("Google検索結果が空でした。")
                return {
                    **state,
                    "raw_search_results": [],
                    "error": current_error,
                }

            print(f"検索結果 {len(search_results_list)} 件取得完了。")
            return {
                **state,
                "raw_search_results": search_results_list,
                "error": current_error,
            }
        except Exception as e:
            error_detail = f"Google検索中にエラーが発生しました: {e}"
            print(error_detail)
            updated_error = (
                f"{current_error}\n{error_detail}" if current_error else error_detail
            )
            return {
                **state,
                "raw_search_results": [],
                "error": updated_error,
            }

    def scrape_and_prepare_context_node(state: AgentState) -> AgentState:
        current_error = state.get("error")
        print("--- ステップ1c: Webスクレイピングとコンテキスト準備 ---")
        search_results_list = state.get("raw_search_results", [])
        scraped_contents = []
        max_content_length_per_page = settings.get(
            "max_content_length_per_page_scrape", 2000
        )

        if not search_results_list:
            print("検索結果が空のため、スクレイピングをスキップします。")
            no_search_results_msg = "関連情報は見つかりませんでした。深い考察を行うには情報が不足しています。"
            return {
                **state,
                "scraped_context": no_search_results_msg,
                "error": current_error,
            }

        print(
            f"検索結果から上位{len(search_results_list)}件のウェブページを読み込んでいます..."
        )
        for i, result in enumerate(search_results_list):
            title = result.get("title", "タイトルなし")
            link = result.get("link")
            snippet = result.get("snippet", "スニペットなし")
            snippet_text = snippet if snippet else "概要なし"

            if link:
                try:
                    print(f"  読み込み中 ({i+1}/{len(search_results_list)}): {link}")
                    loader = WebBaseLoader(
                        web_path=link,
                        requests_kwargs={
                            "timeout": 15,
                            "headers": {
                                "User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +[http://www.google.com/bot.html](http://www.google.com/bot.html))"
                            },
                        },
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
                                "article",
                                "main",
                                "section",
                                "blockquote",
                                "td",
                                "th",
                            ],
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
                        )

                        if page_content_after_transform:
                            shortened_content = page_content_after_transform[
                                :max_content_length_per_page
                            ].strip()
                            if shortened_content:
                                scraped_contents.append(
                                    f"参照元URL: {link}\nタイトル: {title}\n内容の抜粋:\n{shortened_content}"
                                )
                            else:
                                scraped_contents.append(
                                    f"参照元URL: {link}\nタイトル: {title}\n概要: {snippet_text}\n(コンテンツ短縮後、空。スニペット利用)"
                                )
                        else:
                            scraped_contents.append(
                                f"参照元URL: {link}\nタイトル: {title}\n概要: {snippet_text}\n(Transformer適用後コンテンツ空。スニペット利用)"
                            )
                    else:
                        scraped_contents.append(
                            f"参照元URL: {link}\nタイトル: {title}\n概要: {snippet_text}\n(WebBaseLoaderがドキュメント返さず。スニペット利用)"
                        )
                except Exception as e_scrape:
                    print(
                        f"  [詳細エラー] URLからのコンテンツ読み込み/処理エラー: {link} - {e_scrape}"
                    )
                    scraped_contents.append(
                        f"参照元URL: {link}\nタイトル: {title}\n概要: {snippet_text}\n(読み込み/処理エラー: {type(e_scrape).__name__}。スニペット利用)"
                    )
            else:
                scraped_contents.append(
                    f"タイトル: {title}\n概要: {snippet_text} (URLなし)"
                )

        search_context_str = (
            "\n\n===\n\n".join(scraped_contents)
            if scraped_contents
            else "関連性の高いウェブページのコンテンツは見つかりませんでした。"
        )
        print(
            f"ウェブページ読み込み完了。最終的なscraped_contextの文字数: {len(search_context_str)}"
        )
        return {**state, "scraped_context": search_context_str, "error": current_error}

    def generate_structured_article_node(state: AgentState) -> AgentState:
        current_error = state.get("error")
        print("--- ステップ2: 構造化記事生成 (哲学的トーン) ---")
        main_title = state["main_title"]
        subtitles_list = state["subtitles"]
        search_context = state["scraped_context"]

        if not isinstance(subtitles_list, list):
            print(
                f"警告: generate_structured_article_node で subtitles がリストではありません: {subtitles_list}"
            )
            subtitles_for_prompt = (
                f"- 「{str(subtitles_list)}」について考察します。"
                if subtitles_list
                else "- (サブタイトル不明)"
            )
        elif not subtitles_list:
            subtitles_for_prompt = "- (サブタイトルが指定されていません)"
        else:
            subtitles_for_prompt = "\n".join(
                [f"- 「{s}」について考察します。" for s in subtitles_list]
            )

        max_search_context_len = settings.get("max_search_context_for_llm", 18000)
        if len(search_context) > max_search_context_len:
            print(
                f"検索コンテキストが長すぎるため短縮します。元: {len(search_context)}, 短縮後: {max_search_context_len}"
            )
            search_context = search_context[:max_search_context_len]

        try:
            system_template = "あなたは洞察力に優れた哲学者であり、同時に言葉を巧みに操るエッセイストです。与えられた情報から本質を抽出し、読者の知的好奇心を刺激し、深い思索へと誘うような、示唆に富んだ文章を構成してください。あなたの文章は、平易でありながらも深遠な問いを投げかけ、読者自身の内省を促す力を持っています。指定されたJSON形式で、各サブタイトルに対応する考察豊かなコンテンツブロックを作成してください。"
            format_instructions = output_parser.get_format_instructions()
            prompt = ChatPromptTemplate.from_messages(
                [("system", system_template), ("user", GENERATE_ARTICLE_PROMPT_TEXT)]
            ).partial(format_instructions=format_instructions)
            chain = prompt | llm | output_parser
            print("LLMによる哲学的記事生成中...")
            input_data = {
                "main_title": main_title,
                "subtitles": subtitles_for_prompt,
                "search_results": search_context,
            }
            print(
                f"LLMへの入力データ (一部): main_title='{main_title}', subtitles_preview='{subtitles_for_prompt[:100]}...', search_results_len={len(search_context)}"
            )

            parsed_article_obj = chain.invoke(input_data)

            if parsed_article_obj is None:
                raise OutputParserException(
                    "Parsed Article object is None.",
                    llm_output="LLM Response led to None after parsing.",
                )

            generated_article_json = parsed_article_obj.model_dump()
            article_main_title_gen = generated_article_json.get("title", main_title)
            article_blocks_gen = generated_article_json.get("block", [])

            if not article_main_title_gen and not article_blocks_gen:
                error_msg_content = (
                    "LLMが記事のタイトルと本文ブロックの両方を生成できませんでした。"
                )
                updated_error = (
                    f"{current_error}\n記事構造生成エラー: {error_msg_content}"
                    if current_error
                    else f"記事構造生成エラー: {error_msg_content}"
                )
                return {**state, "error": updated_error}

            if isinstance(subtitles_list, list) and len(article_blocks_gen) != len(
                subtitles_list
            ):
                print(
                    f"警告: 生成されたコンテンツブロック数 ({len(article_blocks_gen)}) がサブタイトル数 ({len(subtitles_list)}) と一致しません。"
                )

            article_combined_content_gen = "\n\n".join(article_blocks_gen)
            print(f"哲学的記事生成完了。タイトル: {article_main_title_gen}")
            return {
                **state,
                "generated_article_json": generated_article_json,
                "initial_article_title": article_main_title_gen,
                "initial_article_content": article_combined_content_gen,
                "error": current_error,
            }
        except OutputParserException as e_parse:
            llm_raw_output = (
                e_parse.llm_output if hasattr(e_parse, "llm_output") else "LLM出力不明"
            )
            if llm_raw_output is None:
                llm_raw_output = "LLM returned null/None."
            error_detail = f"記事生成中にOutputParserエラー: {str(e_parse)}\nLLM Raw Output:\n{llm_raw_output}"
            print(error_detail)
            updated_error = (
                f"{current_error}\n{error_detail}" if current_error else error_detail
            )
            return {**state, "error": updated_error}
        except Exception as e:
            error_detail = f"記事生成中に予期せぬ汎用エラー: {e}"
            print(error_detail)
            traceback.print_exc()
            updated_error = (
                f"{current_error}\n{error_detail}" if current_error else error_detail
            )
            return {**state, "error": updated_error}

    def format_html_node(state: AgentState) -> AgentState:
        print("--- ステップ3: HTML整形 (インラインスタイル) ---")
        html_doc_title = state.get("initial_article_title") or state.get(
            "main_title", "生成された考察記事"
        )
        display_main_title_text = state.get(
            "initial_article_title", state.get("main_title", "考察記事")
        )

        subtitles_list = state.get("subtitles", [])
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

        prefecture_image_path_val = state.get("prefecture_image_path")
        current_error_val = state.get("error")

        article_html_parts = []

        # --- インラインスタイル定義 ---
        body_style = "font-family: 'Merriweather', 'Georgia', serif; line-height: 1.8; margin: 0; padding: 0; background-color: #f4f4f0; color: #3a3a3a;"
        container_style = "max-width: 800px; margin: 50px auto; background-color: #ffffff; padding: 40px 50px 50px; border-radius: 3px; box-shadow: 0 10px 30px rgba(0,0,0,0.07); border-top: 6px solid #2c3e50;"
        h1_base_style = "font-family: 'Playfair Display', serif; font-weight: 700; letter-spacing: 0.5px; text-align: center;"
        main_title_style_html = f"{h1_base_style} font-size: 2.6em; color: #1a1a1a; border-bottom: 2px solid #eaeaea; padding-bottom: 20px; margin-top: 0; margin-bottom: 35px;"
        img_container_style = "text-align: center; margin-bottom: 30px;"
        img_style = "max-width: 100%; height: auto; border-radius: 4px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);"
        h2_base_style = "font-family: 'Playfair Display', serif; font-weight: 600;"
        subtitle_style_html = f"{h2_base_style} font-size: 1.9em; color: #1c1c1c; margin-top: 45px; margin-bottom: 20px; border-bottom: 1px solid #f0f0f0; padding-bottom: 12px;"
        p_style_html = "margin-bottom: 1.8em; font-size: 1.1em; color: #3a3a3a; text-align: justify; hyphens: auto; orphans: 2; widows: 2; word-wrap: break-word;"
        error_container_style = "padding: 20px; border: 1px solid #d32f2f; background-color: #ffebee; border-radius: 4px; margin-bottom: 20px;"
        error_title_style_html = f"{h1_base_style} font-size: 1.8em; color: #b71c1c; text-align: left; margin-bottom: 10px;"
        error_message_style_html = (
            f"font-size: 1.0em; color: #c62828; margin-bottom: 0.8em;"
        )
        warning_message_style_html = f"font-size: 1.0em; color: #8c5a00; background-color: #fff8e1; border: 1px solid #ffecb3; padding: 12px; border-radius: 4px; margin-bottom: 1.5em;"
        info_message_style_html = f"font-size: 1.0em; color: #555; margin-bottom: 1em;"
        footer_style_html = "margin-top: 60px; padding-top: 20px; border-top: 1px solid #eaeaea; text-align: center;"
        footer_text_style_html = (
            "font-size: 0.9em; color: #888888; font-style: italic; margin: 0;"
        )
        # --- スタイル定義ここまで ---

        article_html_parts.append(
            f'<h1 style="{main_title_style_html}">{display_main_title_text}</h1>\n'
        )

        if prefecture_image_path_val:
            try:
                with open(prefecture_image_path_val, "rb") as img_file:
                    import base64

                    b64_string = base64.b64encode(img_file.read()).decode()
                    article_html_parts.append(f'<div style="{img_container_style}">')
                    article_html_parts.append(
                        f'  <img src="data:image/png;base64,{b64_string}" alt="Generated image for {display_main_title_text}" style="{img_style}">'
                    )
                    article_html_parts.append(f"</div>\n")
                    print(
                        f"画像をBase64エンコードしてHTMLに埋め込みました: {prefecture_image_path_val}"
                    )
            except Exception as e_img_embed:
                print(
                    f"画像ファイル ({prefecture_image_path_val}) の読み込みまたはBase64エンコードに失敗: {e_img_embed}"
                )
                article_html_parts.append(
                    f'<p style="{error_message_style_html}">画像 ({prefecture_image_path_val}) の表示に失敗しました。</p>\n'
                )

        if current_error_val:
            article_html_parts.append(f'<div style="{error_container_style}">')
            article_html_parts.append(
                f'<h2 style="{error_title_style_html}">処理中に問題が発生しました</h2>'
            )
            error_lines = str(current_error_val).split("\n")
            for line in error_lines:
                if line.strip():
                    article_html_parts.append(
                        f'<p style="{error_message_style_html}">{line}</p>'
                    )
            article_html_parts.append(f"</div>\n")

        if (
            isinstance(subtitles_list, list)
            and subtitles_list
            and article_content_blocks
            and len(article_content_blocks) == len(subtitles_list)
            and not current_error_val
        ):
            for i, subtitle_text in enumerate(subtitles_list):
                article_html_parts.append(
                    f'<h2 style="{subtitle_style_html}">{str(subtitle_text).strip()}</h2>\n'
                )
                content_for_this_subtitle = article_content_blocks[i]
                paragraphs = str(content_for_this_subtitle).strip().split("\n\n")
                for p_content in paragraphs:
                    if p_content.strip():
                        p_content_with_br = p_content.strip().replace("\n", "<br>\n")
                        article_html_parts.append(
                            f'<p style="{p_style_html}">{p_content_with_br}</p>\n'
                        )
        elif state.get("initial_article_content"):
            if not current_error_val:
                article_html_parts.append(
                    f'<p style="{warning_message_style_html}"><strong>注意:</strong> 記事の内部構造が期待通りに生成されなかったため、結合された内容を表示します。</p>'
                )
            else:
                article_html_parts.append(
                    f'<p style="{info_message_style_html}">エラーが発生しましたが、部分的に生成された可能性のあるコンテンツを以下に示します:</p>'
                )

            paragraphs = (
                str(state.get("initial_article_content", "")).strip().split("\n\n")
            )
            for p_content in paragraphs:
                if p_content.strip():
                    p_content_with_br = p_content.strip().replace("\n", "<br>\n")
                    article_html_parts.append(
                        f'<p style="{p_style_html}">{p_content_with_br}</p>\n'
                    )
        elif not current_error_val:
            article_html_parts.append(
                f'<p style="{info_message_style_html}">言葉はまだ紡がれていません。記事コンテンツが生成されませんでした。</p>'
            )

        if not article_html_parts and not current_error_val:
            article_html_parts.append(
                f'<p style="{info_message_style_html}">表示できるコンテンツがありません。</p>'
            )

        article_html_body_content = "".join(article_html_parts)

        final_html_output = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html_doc_title}</title>
    <link href="[https://fonts.googleapis.com/css2?family=Merriweather:ital,wght@0,400;0,700;1,400&family=Playfair+Display:wght@600;700&display=swap](https://fonts.googleapis.com/css2?family=Merriweather:ital,wght@0,400;0,700;1,400&family=Playfair+Display:wght@600;700&display=swap)" rel="stylesheet">
</head>
<body style="{body_style}">
    <div style="{container_style}">
        {article_html_body_content}
        <div style="{footer_style_html}">
            <p style="{footer_text_style_html}">この記事が、あなたの思索の一助となれば幸いです。</p>
        </div>
    </div>
</body>
</html>
"""
        print("HTML整形完了 (インラインスタイル)。")
        return {**state, "html_output": final_html_output}

    workflow = StateGraph(AgentState)
    workflow.add_node("generate_prefecture_image", generate_prefecture_image_node)
    workflow.add_node("generate_search_query", generate_search_query_node)
    workflow.add_node("google_search", google_search_node)
    workflow.add_node("scrape_and_prepare_context", scrape_and_prepare_context_node)
    workflow.add_node("generate_structured_article", generate_structured_article_node)
    workflow.add_node("format_html", format_html_node)

    workflow.set_entry_point("generate_prefecture_image")
    workflow.add_edge("generate_prefecture_image", "generate_search_query")
    workflow.add_edge("generate_search_query", "google_search")
    workflow.add_edge("google_search", "scrape_and_prepare_context")
    workflow.add_edge("scrape_and_prepare_context", "generate_structured_article")
    workflow.add_edge("generate_structured_article", "format_html")
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
        "prefecture_image_path": None,
        "html_output": "",
        "error": None,
    }

    final_state_result = None
    try:
        print("\n--- ワークフロー実行開始 ---")
        final_state_result = app.invoke(initial_state, {"recursion_limit": 20})
        print("--- ワークフロー実行完了 ---")
    except Exception as e_invoke:
        error_msg = f"ワークフロー実行(invoke)中にエラー: {e_invoke}"
        print(error_msg)
        traceback.print_exc()
        current_html_invoke_error = ""
        if isinstance(final_state_result, dict) and "html_output" in final_state_result:
            current_html_invoke_error = final_state_result.get("html_output", "")
        elif isinstance(initial_state, dict) and "html_output" in initial_state:
            current_html_invoke_error = initial_state.get("html_output", "")
        error_html = f"<h1>ワークフロー実行時エラー</h1><p>{error_msg}</p>" + (
            f"<hr><h3>部分的なHTMLコンテンツ:</h3>{current_html_invoke_error}"
            if current_html_invoke_error
            else ""
        )
        return {
            "success": False,
            "main_title": main_title_input,
            "subtitles": subtitles_input,
            "html_output": error_html,
            "error_message": error_msg,
            "final_state_summary": {
                "error": error_msg,
                "last_known_state": (
                    final_state_result if final_state_result else initial_state
                ),
            },
        }

    print("\n--- 全ての処理が完了しました (ワークフロー終了後) ---")
    if final_state_result:
        error_message_from_state = final_state_result.get("error")
        success_status = not bool(error_message_from_state)
        html_content_output = final_state_result.get("html_output", "")
        summary_html_preview = html_content_output
        if len(summary_html_preview) > 300:
            summary_html_preview = summary_html_preview[:300] + "..."

        final_state_summary_dict = {
            k: (
                v[:200] + "..."
                if isinstance(v, str) and len(v) > 200 and k != "html_output_preview"
                else v
            )
            for k, v in final_state_result.items()
            if k not in ["raw_search_results", "scraped_context", "html_output"]
        }
        final_state_summary_dict["html_output_preview"] = (
            summary_html_preview if html_content_output else "(HTMLなし)"
        )

        print(
            f"HTML Outputの先頭100文字: {html_content_output[:100].replace(os.linesep, ' ')}"
            if html_content_output
            else "（HTMLなし）"
        )
        if error_message_from_state:
            print(f"完了時のエラーメッセージ:\n{error_message_from_state}")
        else:
            print("エラーメッセージはありませんでした。")

        return {
            "success": success_status,
            "main_title": main_title_input,
            "subtitles": subtitles_input,
            "html_output": html_content_output,
            "error_message": error_message_from_state,
            "final_state_summary": final_state_summary_dict,
            "prefecture_image_path_final": final_state_result.get(
                "prefecture_image_path"
            ),
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


if __name__ == "__main__":
    print("メイン実行ブロック開始...")
    test_main_title = "東京都"
    test_subtitles = [
        "東京の歴史的変遷とその現代的意義",
        "文化の坩堝としての東京：多様性と創造性",
        "未来都市東京の展望と課題",
    ]
    attempt_flag = True
    print(f"\n記事生成ワークフローを呼び出します (タイトル: '{test_main_title}')...")
    result = generate_article_workflow(
        test_main_title, test_subtitles, attempt_prefecture_image=attempt_flag
    )
    print("\n--- ワークフローからの最終結果 ---")
    print(f"成功ステータス: {result.get('success')}")
    print(f"メインタイトル（入力）: {result.get('main_title')}")
    if result.get("error_message"):
        print(f"エラーメッセージ: {result.get('error_message')}")

    if result.get("prefecture_image_path_final"):
        print(
            f"最終的な都道府県画像パス（状態より）: {result.get('prefecture_image_path_final')}"
        )

    output_html_file = "generated_article_output.html"
    try:
        with open(output_html_file, "w", encoding="utf-8") as f:
            f.write(
                result.get(
                    "html_output", "<p>HTMLコンテンツが生成されませんでした。</p>"
                )
            )
        print(f"HTML出力は '{output_html_file}' に保存されました。")
    except Exception as e_write:
        print(f"HTMLファイルの書き出し中にエラー: {e_write}")
    print("\nメイン実行ブロック終了。")
