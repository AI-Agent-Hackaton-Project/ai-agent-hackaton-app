import streamlit as st
from config.env_config import get_env_config
from langchain_google_vertexai import ChatVertexAI
import json
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from config.prompt_config import GENERATE_ARTICLE_PROMPT
from langchain_core.runnables import RunnableLambda
from typing import List
from pydantic import BaseModel, Field


class ArticleSection(BaseModel):
    heading: str = Field(description="見出し")
    content: str = Field(description="本文")


class Article(BaseModel):
    title: str = Field(description="記事のタイトル")
    block: List[str] = Field(description="記事の各ブロックの本文リスト")


def generate_article(selected_prefecture):
    settings = get_env_config()

    llm = ChatVertexAI(
        model=settings["model_name"],
        temperature=0,
        max_tokens=None,
        max_retries=6,
        stop=None,
    )

    system_template = "あなたはプロのWEBライターです。"

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", GENERATE_ARTICLE_PROMPT)]
    )

    chain = (
        {
            "selected_prefecture": itemgetter("selected_prefecture"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    input_data = {
        "selected_prefecture": selected_prefecture,
    }

    try:
        response_str = chain.invoke(input_data)
        return json.loads(response_str)
    except json.JSONDecodeError:
        print(
            "⚠️ JSON パースに失敗しました。LLMの返答が JSON 形式ではない可能性があります。"
        )
        print("--- LLM Raw Response ---")
        print(response_str)
        print("--- End LLM Raw Response ---")
        return {"error": "JSON Decode Error", "raw_response": response_str}
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        import traceback

        traceback.print_exc()
        return {"error": "Unexpected Error", "details": str(e)}
