import streamlit as st
from config.env_config import get_env_config
from langchain_google_vertexai import ChatVertexAI

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
import traceback

from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate
from prompts.prompt_config import (
    GENERATE_ARTICLE_PROMPT,
)
from typing import List
from pydantic import BaseModel, Field


class Article(BaseModel):
    title: str = Field(description="記事のタイトル")
    block: List[str] = Field(description="記事の各ブロックの本文リスト")


def generate_article(selected_prefecture: str) -> dict:
    settings = get_env_config()

    llm = ChatVertexAI(
        model_name=settings.get("model_name", "gemini-pro"),
        temperature=0,
        max_output_tokens=settings.get("max_output_tokens", 4096),
        max_retries=6,
        stop=None,
    )

    output_parser = PydanticOutputParser(pydantic_object=Article)

    system_template = "あなたはプロのWEBライターです。依頼された形式に従って、魅力的で正確な情報に基づいた記事を作成してください。"

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", GENERATE_ARTICLE_PROMPT)]
    ).partial(format_instructions=output_parser.get_format_instructions())

    chain = (
        {
            "selected_prefecture": itemgetter("selected_prefecture"),
        }
        | prompt
        | llm
        | output_parser
    )

    input_data = {
        "selected_prefecture": selected_prefecture,
    }

    try:
        parsed_article: Article = chain.invoke(input_data)
        # Pydantic V2 を想定: .model_dump()
        # Pydantic V1 の場合: .dict()
        return parsed_article.model_dump()

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
        }

    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        traceback.print_exc()
        return {"error": "Unexpected Error", "details": str(e)}
