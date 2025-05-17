import streamlit as st
from config.env_config import get_env_config
from langchain_google_vertexai import ChatVertexAI

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
import traceback

from langchain_core.prompts import ChatPromptTemplate
from prompts.GENERATE_ARTICLE import (  # パスが正しいか確認
    GENERATE_ARTICLE_PROMPT,
)
from typing import List
from pydantic import BaseModel, Field

from langchain_google_community.search import (
    GoogleSearchAPIWrapper,
)

# WebBaseLoaderをインポート
from langchain_community.document_loaders import WebBaseLoader

# BeautifulSoupTransformerをインポート (オプションの前処理用)
from langchain_community.document_transformers import BeautifulSoupTransformer

import json


class Article(BaseModel):
    title: str = Field(description="記事のタイトル")
    block: List[str] = Field(description="記事の各ブロックの本文リスト")


def generate_article(selected_prefecture: str) -> dict:
    settings = get_env_config()

    google_api_key = settings.get("google_api_key")
    google_cse_id = settings.get("google_cse_id")

    if not google_api_key or not google_cse_id:
        st.error("Google APIキーまたはCSE IDが設定されていません。")
        return {
            "error": "Search Configuration Error",
            "details": "Google API Key or CSE ID is missing.",
        }

    search_results_str = ""
    raw_search_results_for_display = []  # UI表示用に生の検索結果リストも保持

    try:
        st.write(f"「{selected_prefecture}」に関する情報をGoogleで検索中...")
        search_wrapper = GoogleSearchAPIWrapper(
            google_api_key=google_api_key,
            google_cse_id=google_cse_id,
            # dateRestrict="m6" # コンストラクタではなくresultsメソッドのkwargsで渡すのが推奨
        )
        search_query = f"{selected_prefecture} 歴史 最新情報 観光"
        num_pages_to_scrape = settings.get(
            "num_pages_to_scrape", 2
        )  # スクレイピングするページ数 (設定ファイルやデフォルト値)

        search_results_list = search_wrapper.results(
            query=search_query,
            num_results=num_pages_to_scrape,  # スクレイピング対象のURLを取得する件数
            dateRestrict="m6",
        )

        raw_search_results_for_display = search_results_list[:]  # UI表示用にコピー

        print(f"--- 「{search_query}」の検索結果 (resultsメソッド) ---")
        if search_results_list:
            scraped_contents = []
            st.write(
                f"検索結果から上位{len(search_results_list)}件のウェブページを読み込んでいます..."
            )
            for i, result in enumerate(search_results_list):
                title = result.get("title", "タイトルなし")
                link = result.get("link")
                snippet = result.get("snippet", "スニペットなし")
                print(f"\n結果 {i+1}:")
                print(f"  タイトル: {title}")
                print(f"  リンク: {link}")
                print(f"  スニペット: {snippet}")

                if link:
                    try:
                        # 各URLのコンテンツをロード
                        st.write(f"  読み込み中: {link}")
                        loader = WebBaseLoader(web_path=link)
                        # タイムアウトやエラーハンドリングを追加することも検討
                        # loader.requests_per_second = 1 # Polite scraping
                        # loader.requests_kwargs = {'timeout': 10}
                        documents = loader.load()  # Documentオブジェクトのリストが返る

                        if documents:
                            # BeautifulSoupTransformerでHTMLをクリーニング (オプション)
                            # 特定のタグのみ抽出するなどの設定が可能
                            # bs_transformer = BeautifulSoupTransformer()
                            # docs_transformed = bs_transformer.transform_documents(documents, tags_to_extract=["p", "h1", "h2", "h3", "li"])
                            # page_content = " ".join([doc.page_content for doc in docs_transformed])

                            # シンプルに全テキストを取得 (長さに注意)
                            page_content = " ".join(
                                [doc.page_content for doc in documents]
                            )

                            # コンテキスト長を考慮してコンテンツを短縮 (例: 先頭N文字)
                            max_content_length_per_page = settings.get(
                                "max_content_length_per_page", 2000
                            )
                            shortened_content = page_content[
                                :max_content_length_per_page
                            ]

                            scraped_contents.append(
                                f"参照元URL: {link}\nタイトル: {title}\n内容:\n{shortened_content}"
                            )
                            print(
                                f"    -> コンテンツ取得成功 (先頭{len(shortened_content)}文字)"
                            )
                        else:
                            print(
                                f"    -> コンテンツ取得失敗 (ドキュメントなし): {link}"
                            )
                    except Exception as e_scrape:
                        print(
                            f"    -> URLからのコンテンツ読み込みエラー: {link}, エラー: {e_scrape}"
                        )
                        # エラーが発生した場合でも、スニペットは利用可能なら含める
                        scraped_contents.append(
                            f"参照元URL: {link}\nタイトル: {title}\n概要: {snippet}\n(コンテンツの読み込みに失敗しました)"
                        )

            if scraped_contents:
                search_results_str = "\n\n===\n\n".join(scraped_contents)
            else:
                search_results_str = (
                    "関連性の高いウェブページのコンテンツは見つかりませんでした。"
                )
        else:
            print("検索結果が見つかりませんでした。")
            search_results_str = "関連情報は見つかりませんでした。"

        st.write("検索およびウェブページ読み込み完了。")
    except Exception as e:
        st.warning(
            f"Google検索またはウェブページ読み込み中にエラーが発生しました: {e}. 検索なしで記事を生成します。"
        )
        search_results_str = "検索またはウェブページ読み込み中にエラーが発生したため、追加情報はありません。"
        traceback.print_exc()

    # --- LLM設定と記事生成部分は変更なし ---
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
        "search_results": search_results_str,  # ここにスクレイピングした内容が入る
    }

    try:
        st.write("LLMによる記事生成中...")
        parsed_article: Article = chain.invoke(input_data)
        st.write("記事生成完了。")
        return {
            "article_content": parsed_article.model_dump(),
            # search_results_for_display には整形前の検索結果リスト(タイトル/リンク/スニペット)を渡す
            "search_results_for_display": raw_search_results_for_display,
        }

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
            "search_results_for_display": raw_search_results_for_display,
        }

    except Exception as e:
        print(f"予期せぬエラー: {e}")
        traceback.print_exc()
        return {
            "error": "Unexpected Error",
            "details": str(e),
            "search_results_for_display": raw_search_results_for_display,
        }
