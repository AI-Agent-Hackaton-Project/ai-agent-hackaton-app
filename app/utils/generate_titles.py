import streamlit as st
from config.env_config import get_env_config  # 仮定: 存在する設定ファイル
from langchain_google_vertexai import ChatVertexAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
import traceback
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any, Union  # Union を追加
from pydantic import BaseModel, Field

from langchain_google_community.search import (
    GoogleSearchAPIWrapper,
)
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
import json


# --- 新しいPydanticモデル定義 ---
class TitlesOutput(BaseModel):
    main_title: str = Field(
        description="生成されたメインタイトル。必ず30文字以上で、指定された都道府県名を含むこと。"
    )
    sub_titles: List[str] = Field(
        description="生成されたサブタイトルのリスト。5つのサブタイトルを含むこと。",
        min_items=5,
        max_items=5,
    )


PHILOSOPHICAL_TITLES_PROMPT_TEMPLATE = """
あなたは、場所の持つ意味や物語を掘り下げ、読者に哲学的な問いを投げかける思索家であり、エッセイストです。
提供された情報と都道府県名に基づき、人々の知的好奇心を刺激し、内省を促すようなタイトル群を生成してください。
今回のコンセプトは「地図の中の哲学者」です。生成するタイトルは、このコンセプトを色濃く反映するものとします。

# コンセプト「地図の中の哲学者」について:
- 地図上の場所（今回は {selected_prefecture}）を選ぶと、その場所の歴史・文化・自然・社会背景（例：過疎化、伝統技術の継承、災害からの復興など）を深く掘り下げます。
- 単なる情報提供ではなく、その場所が持つ「意味」や、私たち自身への「問い」を浮き彫りにし、読者が「場所をきっかけに自分を問い直す」ような体験を促すことを目指します。

# 都道府県名:
{selected_prefecture}

# 提供された検索結果（コンテキスト）:
{search_results}

# 指示:
提供された検索結果（コンテキスト）と上記の都道府県名「{selected_prefecture}」に基づいて、以下の条件と「地図の中の哲学者」のコンセプトを深く理解した上で、メインタイトル1つとサブタイトル5つを生成してください。

- **メインタイトル:**
    - SEOも意識しつつ、**{selected_prefecture} の本質に迫るような、あるいはその土地から発せられる哲学的な問いを投げかけるような、思索的で魅力的なタイトル**にしてください。
    - 例：「{selected_prefecture}の黄昏に聴く、変わりゆく故郷と魂の響き」（←この例も30字以内に収まるように調整するとより良いでしょう）
    - **最重要条件:** メインタイトルは**必ず20文字以上、30文字以内**にしてください。文字数が足りない場合や長すぎる場合は、場所の持つ意味、問いかけ、感情、風景描写などを調整し、**必ずこの範囲の長さに**してください。この文字数範囲外のタイトルは許可されません。
    - **必ず「{selected_prefecture}」というキーワードを最低1回は含めてください**。

- **サブタイトル:**
    - 「{selected_prefecture}」や、その土地の歴史、文化、自然、社会背景（例：過疎化、伝統、災害、再生、人々の営みなど）に関連するキーワードを効果的に含めてください。
    - 各サブタイトルは、具体的な情報を示唆しつつ、**読者がその場所の意味について深く考え、自らの価値観や生き方について問い直すきっかけとなるような、哲学的で示唆に富んだ問いかけや視点**を提示してください。
    - **各サブタイトルは、おおむね15文字程度の簡潔さで、読者の興味を引き、深い思索を促す問いを投げかけるものにしてください。**
    - サブタイトル1は「{selected_prefecture}とは？風景が紡ぐ物語」のような、場所の導入と問いかけを簡潔に含む形式にしてください。（おおむね15文字程度）
    - 残りのサブタイトルも、それぞれが独立した問いや思索のテーマを簡潔に持つように工夫してください。
    - 例 (それぞれ15文字程度で):
        - 「〇〇（地名等）に何を聴く？」
        - 「{selected_prefecture}の古道、未来への問い」
        - 「伝統と革新、豊かさの行方」
        - 「旅人よ、{selected_prefecture}で何を見出す？」
        - 「静寂が語る{selected_prefecture}の心」

必ず以下の最終出力の形式に厳密に従ってください。

## **【最終出力の形式】**
{format_instructions}

# あなたが生成する「{selected_prefecture}」に関する「地図の中の哲学者」としてのタイトル群:
"""


def _get_search_results(
    query: str, api_key: str, cse_id: str, num_results: int
) -> List[Dict[str, Any]]:
    """Google検索を実行し、結果のリストを返す。"""
    st.write(f"Google検索中 (クエリ: {query}, {num_results}件)...")
    search_wrapper = GoogleSearchAPIWrapper(
        google_api_key=api_key,
        google_cse_id=cse_id,
    )
    search_results_list = search_wrapper.results(
        query=query,
        num_results=num_results,
    )
    print(f"--- 「{query}」の検索結果 (resultsメソッド) ---")
    if search_results_list:
        for i, result in enumerate(search_results_list):
            print(f"\n結果 {i+1}:")
            print(f"  タイトル: {result.get('title')}")
            print(f"  リンク: {result.get('link')}")
            print(f"  スニペット: {result.get('snippet')}")
    else:
        print("検索結果が見つかりませんでした。")
    st.write("Google検索完了。")
    return search_results_list if search_results_list else []


def _scrape_and_prepare_context(
    search_results_list: List[Dict[str, Any]], settings: dict
) -> str:
    """検索結果のURLからウェブページをスクレイピングし、LLM用コンテキスト文字列を作成する。"""
    scraped_contents = []
    if not search_results_list:
        return "関連情報は見つかりませんでした。"

    st.write(
        f"検索結果から上位{len(search_results_list)}件のウェブページを読み込んでいます..."
    )
    max_content_length_per_page = settings.get("max_content_length_per_page", 2000)

    for i, result in enumerate(search_results_list):
        title = result.get("title", "タイトルなし")
        link = result.get("link")
        snippet = result.get("snippet", "スニペットなし")

        if link:
            try:
                st.write(f"  読み込み中 ({i+1}/{len(search_results_list)}): {link}")
                loader = WebBaseLoader(
                    web_path=link, requests_kwargs={"timeout": 10}
                )  # タイムアウト設定
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
                            f"    -> コンテンツ取得成功 (先頭{len(shortened_content)}文字): {link}"
                        )
                    else:
                        scraped_contents.append(
                            f"参照元URL: {link}\nタイトル: {title}\n概要: {snippet}\n(主要コンテンツ抽出失敗)"
                        )
                        print(f"    -> 主要コンテンツ抽出失敗、スニペット利用: {link}")

                else:
                    scraped_contents.append(
                        f"参照元URL: {link}\nタイトル: {title}\n概要: {snippet}\n(コンテンツ取得失敗: ドキュメントなし)"
                    )
                    print(f"    -> コンテンツ取得失敗 (ドキュメントなし): {link}")
            except Exception as e_scrape:
                print(
                    f"    -> URLからのコンテンツ読み込みエラー: {link}, エラー: {e_scrape}"
                )
                scraped_contents.append(
                    f"参照元URL: {link}\nタイトル: {title}\n概要: {snippet}\n(コンテンツの読み込みに失敗しました: {type(e_scrape).__name__})"
                )
        else:
            scraped_contents.append(f"タイトル: {title}\n概要: {snippet} (URLなし)")

    if scraped_contents:
        search_context_str = "\n\n===\n\n".join(scraped_contents)
    else:
        search_context_str = (
            "関連性の高いウェブページのコンテンツは見つかりませんでした。"
        )

    st.write("ウェブページ読み込み完了。")
    return search_context_str


# --- LLMチェーン関数の修正 ---
def _invoke_llm_for_titles(
    selected_prefecture: str, search_context: str, settings: dict
) -> Union[TitlesOutput, dict]:
    """LLMチェーンを準備・実行し、パースされたタイトル群またはエラー情報を返す。"""
    llm = ChatVertexAI(
        model_name=settings.get("model_name", "gemini-1.0-pro-001"),
        temperature=0,
        max_output_tokens=settings.get(
            "max_output_tokens", 2048
        ),  # タイトルなので元の8192も不要かも
        max_retries=settings.get(
            "llm_max_retries", 6
        ),  # settingsから取得するように変更
        stop=None,
    )
    output_parser = PydanticOutputParser(
        pydantic_object=TitlesOutput
    )  # 新しいPydanticモデル
    system_template = "あなたはプロのWEBライターです。提供された検索結果と指示に基づいて、指定されたJSON形式でSEOを考慮した魅力的なメインタイトルとサブタイトルを生成してください。"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            ("user", PHILOSOPHICAL_TITLES_PROMPT_TEMPLATE),
        ]
    ).partial(format_instructions=output_parser.get_format_instructions())

    chain = prompt | llm | output_parser
    input_data = {
        "selected_prefecture": selected_prefecture,
        "search_results": search_context,
    }

    try:
        st.write("LLMによるタイトル生成中...")
        parsed_titles: TitlesOutput = chain.invoke(input_data)
        st.write("タイトル生成完了。")
        return parsed_titles  # Pydanticモデルのインスタンスを返す
    except OutputParserException as e:
        llm_output = getattr(e, "llm_output", str(e.args[0] if e.args else str(e)))
        print(f"⚠️ PydanticOutputParser Error: {e}")
        print(
            f"--- LLM Raw Response (OutputParserException) ---\n{llm_output}\n--- End ---"
        )
        return {
            "error": "Output Parser Error",
            "raw_response": llm_output,
            "details": str(e),
        }
    except Exception as e:
        print(f"LLM呼び出し中に予期せぬエラーが発生しました: {e}")
        traceback.print_exc()
        return {
            "error": "LLM Invocation Error",
            "details": str(e),
        }


def generate_titles_for_prefecture(selected_prefecture: str) -> dict:
    """指定された都道府県に関するメインタイトルとサブタイトルを生成する。"""
    settings = get_env_config()

    google_api_key = settings.get("google_api_key")
    google_cse_id = settings.get("google_cse_id")

    if not google_api_key or not google_cse_id:
        st.error("Google APIキーまたはCSE IDが設定されていません。")
        return {
            "error": "Search Configuration Error",
            "details": "Google API Key or CSE ID is missing.",
            "search_results_for_display": [],
            "titles_output": None,
        }

    raw_search_results_for_display = []
    search_context_str = "検索処理が実行されませんでした。"

    try:
        num_pages_to_scrape = settings.get("search_num_results", 10)
        search_query = f"{selected_prefecture} 観光"
        raw_search_results_for_display = _get_search_results(
            search_query,
            google_api_key,
            google_cse_id,
            num_pages_to_scrape,
        )

        if not raw_search_results_for_display:
            search_context_str = "関連情報は見つかりませんでした。"
        else:
            search_context_str = _scrape_and_prepare_context(
                raw_search_results_for_display, settings
            )

    except Exception as e_search_scrape:
        st.warning(
            f"Google検索またはウェブページ読み込み中にエラーが発生しました: {e_search_scrape}. 検索コンテキストなしでタイトル生成を試みます。"
        )
        search_context_str = "検索またはウェブページ読み込み中にエラーが発生したため、追加情報はありません。"
        traceback.print_exc()

    # LLMによるタイトル生成
    llm_response_or_titles = _invoke_llm_for_titles(  # 修正された関数を呼び出し
        selected_prefecture, search_context_str, settings
    )

    # 最終的な戻り値を組み立て
    if isinstance(
        llm_response_or_titles, TitlesOutput
    ):  # 成功時はPydanticモデルのインスタンス
        return {
            "titles_output": llm_response_or_titles.model_dump(),  # Pydanticモデルを辞書に変換
            "search_results_for_display": raw_search_results_for_display,
        }
    elif (
        isinstance(llm_response_or_titles, dict) and "error" in llm_response_or_titles
    ):  # エラー時は辞書
        return {
            **llm_response_or_titles,
            "search_results_for_display": raw_search_results_for_display,
            "titles_output": None,  # エラー時もキーを統一
        }
    else:  # 予期せぬ応答形式
        st.error("LLMからの応答が予期せぬ形式でした。")
        return {
            "error": "Unexpected LLM Response",
            "details": "The response from the LLM was not in the expected format.",
            "search_results_for_display": raw_search_results_for_display,
            "titles_output": None,
        }
