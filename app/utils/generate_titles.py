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


# --- 新しいプロンプトテンプレート定義 ---
GENERATE_TITLES_PROMPT_TEMPLATE = """
あなたはプロのWEBライターです。
提供された検索結果（コンテキスト）と都道府県名に基づいて、SEOを考慮したメインタイトル1つと、それに関連するサブタイトル5つを生成してください。

# 都道府県名:
{selected_prefecture}

# 提供された検索結果（コンテキスト）:
{search_results}

# 指示:
提供された検索結果（コンテキスト）と上記の都道府県名「{selected_prefecture}」に基づいて、以下の条件を満たすメインタイトル1つとサブタイトル5つを生成してください。

- **メインタイトル:**
    - SEOを考慮し、読者の興味を引く魅力的なものにしてください。
    - **最重要条件:** メインタイトルは**絶対に30文字以上**にしてください。文字数が足りない場合は、関連情報や説明、都道府県名を追加するなどして、**必ず30文字を超える長さに**調整してください。30文字未満のタイトルは許可されません。
    - **必ず「{selected_prefecture}」というキーワードを最低1回は含めてください**。

- **サブタイトル:**
    - 「{selected_prefecture}」や関連キーワード（例: 観光、歴史、魅力、最新情報など）をできるだけ含め、記事の内容が推測できるようなものにしてください。
    - サブタイトル1は「{selected_prefecture}とは？魅力や基本情報を網羅解説」のような形式にしてください。
    - 各サブタイトルは具体的で、読者がクリックしたくなるような内容にしてください。

必ず以下の最終出力の形式に厳密に従ってください。

## **【最終出力の形式】**
{format_instructions}

# あなたが生成する「{selected_prefecture}」に関するタイトル群:
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
            ("user", GENERATE_TITLES_PROMPT_TEMPLATE),
        ]  # 新しいプロンプト
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


# --- メイン関数の修正 ---
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
            "titles_output": None,  # エラー時もキーを統一
        }

    raw_search_results_for_display = []
    search_context_str = "検索処理が実行されませんでした。"

    try:
        num_pages_to_scrape = settings.get("num_pages_to_scrape", 10)
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
