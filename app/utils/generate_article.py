import streamlit as st
from config.env_config import get_env_config
from langchain_google_vertexai import ChatVertexAI

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
import traceback

from langchain_core.prompts import ChatPromptTemplate
from prompts.GENERATE_ARTICLE import (  # 正しいパスか確認してください (例: prompts.generate_article_prompt)
    GENERATE_ARTICLE_PROMPT,
)
from typing import List
from pydantic import BaseModel, Field

from langchain_google_community.search import (
    GoogleSearchAPIWrapper,
)


class Article(BaseModel):
    title: str = Field(description="記事のタイトル")
    block: List[str] = Field(description="記事の各ブロックの本文リスト")


def generate_article(selected_prefecture: str) -> dict:
    settings = get_env_config()

    google_api_key = settings.get("google_api_key")
    google_cse_id = settings.get("google_cse_id")

    if not google_api_key or not google_cse_id:
        st.error(
            "Google APIキーまたはCSE IDが設定されていません。検索機能は利用できません。"
        )
        return {
            "error": "Search Configuration Error",
            "details": "Google API Key or CSE ID is missing.",
        }

    search_results_str = ""
    try:
        st.write(f"「{selected_prefecture}」に関する情報をGoogleで検索中...")

        search_wrapper = GoogleSearchAPIWrapper(
            google_api_key=google_api_key, google_cse_id=google_cse_id
        )
        search_query = f"{selected_prefecture} 歴史 最新情報 観光"

        # .run() は検索結果を文字列として返します。デフォルトは10件程度のサマリー。
        search_results_str = search_wrapper.run(search_query)

        if not search_results_str:
            search_results_str = "関連情報は見つかりませんでした。"

        st.write("検索完了。")
    except Exception as e:
        st.warning(
            f"Google検索中にエラーが発生しました: {e}. 検索なしで記事を生成します。"
        )
        search_results_str = "検索中にエラーが発生したため、追加情報はありません。"
        traceback.print_exc() 

    llm = ChatVertexAI(
        model_name=settings.get("model_name", "gemini-1.0-pro-001"),
        temperature=0,
        max_output_tokens=settings.get("max_output_tokens", 8192),
        max_retries=6,
        stop=None,
    )

    output_parser = PydanticOutputParser(pydantic_object=Article)
    system_template = "あなたはプロのWEBライターです。提供された検索結果と指示に基づいて、魅力的で正確な情報に基づいた記事を指定されたJSON形式で作成してください。"
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", GENERATE_ARTICLE_PROMPT)]
    ).partial(format_instructions=output_parser.get_format_instructions())

    chain = prompt | llm | output_parser
    input_data = {
        "selected_prefecture": selected_prefecture,
        "search_results": search_results_str,
    }

    try:
        st.write("LLMによる記事生成中...")
        parsed_article: Article = chain.invoke(input_data)
        st.write("記事生成完了。")
        return {
            "article_content": parsed_article.model_dump(),
            "search_results_for_display": search_results_str,
        }

    except OutputParserException as e:
        llm_output = getattr(e, "llm_output", str(e.args[0] if e.args else str(e)))
        print(
            f"⚠️ PydanticOutputParserによるパースに失敗しました。LLMの返答が期待する形式ではない可能性があります。\nエラー: {e}"
        )
        print("--- LLM Raw Response (at OutputParserException) ---")
        print(llm_output)
        print("--- End LLM Raw Response ---")
        return {
            "error": "Output Parser Error",
            "raw_response": llm_output,
            "details": str(e),
            "search_results_for_display": search_results_str,
        }

    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        traceback.print_exc()
        return {
            "error": "Unexpected Error",
            "details": str(e),
            "search_results_for_display": search_results_str,
        }
