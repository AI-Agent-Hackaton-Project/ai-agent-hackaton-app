import streamlit as st
from config.env_config import get_env_config
from langchain_google_vertexai import ChatVertexAI

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
import traceback

from langchain_core.prompts import ChatPromptTemplate
from prompts.GENERATE_ARTICLE import (
    GENERATE_ARTICLE_PROMPT,
)
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from langchain_google_community.search import (
    GoogleSearchAPIWrapper,
)
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
import json


class Article(BaseModel):
    title: str = Field(description="記事のタイトル")
    block: List[str] = Field(description="記事の各ブロックの本文リスト")


# --- ヘルパー関数定義 ---


def _get_search_results(
    query: str, api_key: str, cse_id: str, num_results: int
) -> List[Dict[str, Any]]:
    """Google検索を実行し、結果のリストを返す。"""
    st.write(f"Google検索中 (クエリ: {query}, {num_results}件...")
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
            # リンクがない場合はスニペットのみ利用
            scraped_contents.append(f"タイトル: {title}\n概要: {snippet} (URLなし)")

    if scraped_contents:
        search_context_str = "\n\n===\n\n".join(scraped_contents)
    else:
        search_context_str = (
            "関連性の高いウェブページのコンテンツは見つかりませんでした。"
        )

    st.write("ウェブページ読み込み完了。")
    return search_context_str


def _invoke_llm_chain(
    selected_prefecture: str, search_context: str, settings: dict
) -> dict:
    """LLMチェーンを準備・実行し、パースされた記事またはエラー情報を返す。"""
    llm = ChatVertexAI(
        model_name=settings.get("model_name", "gemini-1.0-pro-001"),
        temperature=0,
        max_output_tokens=settings.get("max_output_tokens", 8192),
        max_retries=6,
        stop=None,
    )
    output_parser = PydanticOutputParser(pydantic_object=Article)
    system_template = "あなたはプロのWEBライターです。提供された検索結果(ウェブページからの抜粋情報)と指示に基づいて、魅力的で正確な情報に基づいた記事を指定されたJSON形式で作成してください。"
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", GENERATE_ARTICLE_PROMPT)]
    ).partial(format_instructions=output_parser.get_format_instructions())

    chain = prompt | llm | output_parser
    input_data = {
        "selected_prefecture": selected_prefecture,
        "search_results": search_context,
    }

    try:
        st.write("LLMによる記事生成中...")
        parsed_article: Article = chain.invoke(input_data)
        st.write("記事生成完了。")
        return parsed_article.model_dump()
    except OutputParserException as e:
        llm_output = getattr(e, "llm_output", str(e.args[0] if e.args else str(e)))
        print(f"⚠️ PydanticOutputParser Error: {e}")
        print(
            f"--- LLM Raw Response (OutputParserException) ---\n{llm_output}\n--- End ---"
        )
        return {  # エラー情報を返す
            "error": "Output Parser Error",
            "raw_response": llm_output,
            "details": str(e),
        }
    except Exception as e:  # LLM呼び出し中のその他のエラー
        print(f"LLM呼び出し中に予期せぬエラーが発生しました: {e}")
        traceback.print_exc()
        return {
            "error": "LLM Invocation Error",
            "details": str(e),
        }


# --- メイン関数 ---


def generate_article(selected_prefecture: str) -> dict:
    settings = get_env_config()

    google_api_key = settings.get("google_api_key")
    google_cse_id = settings.get("google_cse_id")

    if not google_api_key or not google_cse_id:
        st.error("Google APIキーまたはCSE IDが設定されていません。")
        return {
            "error": "Search Configuration Error",
            "details": "Google API Key or CSE ID is missing.",
            "search_results_for_display": [],  # エラー時もキーを統一
        }

    raw_search_results_for_display = []
    search_context_str = "検索処理が実行されませんでした。"  # 初期値

    try:
        num_pages_to_scrape = settings.get("num_pages_to_scrape", 2)

        search_query = f"{selected_prefecture} 歴史 最新情報 観光"
        raw_search_results_for_display = _get_search_results(
            search_query,
            google_api_key,
            google_cse_id,
            num_pages_to_scrape,
        )

        if not raw_search_results_for_display:  # 検索結果が空の場合
            search_context_str = "関連情報は見つかりませんでした。"
        else:
            search_context_str = _scrape_and_prepare_context(
                raw_search_results_for_display, settings
            )

    except Exception as e_search_scrape:
        st.warning(
            f"Google検索またはウェブページ読み込み中にエラーが発生しました: {e_search_scrape}. 検索なしで記事を生成します。"
        )
        search_context_str = "検索またはウェブページ読み込み中にエラーが発生したため、追加情報はありません。"
        traceback.print_exc()

    # LLMによる記事生成
    llm_response_data = _invoke_llm_chain(
        selected_prefecture, search_context_str, settings
    )

    # 最終的な戻り値を組み立て
    if "error" in llm_response_data:
        return {
            **llm_response_data,  # LLMからのエラー情報
            "search_results_for_display": raw_search_results_for_display,
        }
    else:
        return {
            "article_content": llm_response_data,
            "search_results_for_display": raw_search_results_for_display,
        }
