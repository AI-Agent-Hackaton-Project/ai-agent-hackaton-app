from typing import Dict, Any, List
from langchain_google_vertexai import ChatVertexAI
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_community.search import GoogleSearchAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_transformers import BeautifulSoupTransformer

from config.env_config import get_env_config
from prompts.GENERATE_ARTICLE_PROMPT_TEXT import GENERATE_ARTICLE_PROMPT_TEXT
from prompts.APHORISM_PROMPT_TEXT import APHORISM_PROMPT_TEXT
from utils.generate_four_images import generate_four_images
from utils.generate_titles_images import generate_prefecture_image_and_get_path
from .html_formatter import build_html_article

from pydantic import BaseModel, Field


class Article(BaseModel):
    title: str = Field(description="記事のタイトル (メインタイトル)")
    block: List[str] = Field(description="記事の各ブロックの本文リスト")


def generate_search_query(state: Dict[str, Any]) -> Dict[str, Any]:
    """検索クエリを生成"""
    subtitles_str = " ".join(state.get("subtitles", []))
    state["search_query"] = f"{state['main_title']} {subtitles_str} 解説 歴史 哲学"
    return state


def perform_google_search(state: Dict[str, Any]) -> Dict[str, Any]:
    """Google検索を実行"""
    try:
        settings = get_env_config()
        google_api_key = settings.get("google_api_key")
        google_cse_id = settings.get("google_cse_id")

        if not google_api_key or not google_cse_id:
            raise ValueError("Google APIキーまたはCSE IDが設定されていません。")

        search_wrapper = GoogleSearchAPIWrapper(
            google_api_key=google_api_key, google_cse_id=google_cse_id
        )
        results = search_wrapper.results(query=state["search_query"], num_results=5)
        state["raw_search_results"] = results
    except Exception as e:
        state["error"] = f"Google検索エラー: {e}"

    return state


def scrape_and_prepare_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """ウェブページをスクレイピングしてコンテキストを準備"""
    results = state.get("raw_search_results", [])
    if not results:
        state["scraped_context"] = "関連情報が見つかりませんでした。"
        return state

    contents = []
    for res in results:
        try:
            docs = WebBaseLoader(
                web_path=res["link"], requests_kwargs={"timeout": 10}
            ).load()
            transformed = BeautifulSoupTransformer().transform_documents(
                docs, tags_to_extract=["p", "h2", "h3"]
            )
            content = " ".join([d.page_content for d in transformed]).strip()
            if content:
                contents.append(f"参照元: {res['link']}\n内容: {content[:1500]}")
        except Exception:
            continue

    state["scraped_context"] = "\n\n---\n\n".join(contents) or "ウェブ情報取得不可"
    return state


def generate_article_content(state: Dict[str, Any]) -> Dict[str, Any]:
    """記事本文を生成"""
    try:
        settings = get_env_config()
        llm = ChatVertexAI(
            model_name=settings.get("model_name", "gemini-1.5-pro-001"),
            project=settings.get("gcp_project_id"),
            location=settings.get("gcp_location"),
            temperature=settings.get("temperature", 0.7),
            max_output_tokens=settings.get("max_output_tokens", 8192),
        )

        output_parser = PydanticOutputParser(pydantic_object=Article)
        chain = (
            ChatPromptTemplate.from_template(GENERATE_ARTICLE_PROMPT_TEXT)
            | llm
            | output_parser
        )

        article = chain.invoke(
            {
                "format_instructions": output_parser.get_format_instructions(),
                "search_results": state["scraped_context"],
                "main_title": state["main_title"],
                "subtitles": "\n- ".join(state["subtitles"]),
            }
        )

        state["generated_article_json"] = article.model_dump()
        state["initial_article_title"] = article.title

    except Exception as e:
        state["error"] = f"記事生成エラー: {e}"

    return state


def generate_aphorism(state: Dict[str, Any]) -> Dict[str, Any]:
    """名言を生成"""
    try:
        settings = get_env_config()
        llm = ChatVertexAI(
            model_name=settings.get("model_name", "gemini-1.5-pro-001"),
            project=settings.get("gcp_project_id"),
            location=settings.get("gcp_location"),
            temperature=0.8,
        )

        prompt = ChatPromptTemplate.from_template(APHORISM_PROMPT_TEXT)
        chain = prompt | llm | StrOutputParser()
        aphorism = chain.invoke({"region": state["main_title"]})
        state["aphorism"] = aphorism.strip()

    except Exception as e:
        state["error"] = f"名言生成エラー: {e}"

    return state


def generate_main_image(
    state: Dict[str, Any],
    attempt_prefecture_image: bool = True,
) -> Dict[str, Any]:
    """4コマ画像を生成"""
    if not attempt_prefecture_image:
        state["main_theme_image_path"] = None
        return state

    try:
        image_path = generate_four_images(state["selected_prefecture_name"])
        state["main_theme_image_path"] = image_path
    except Exception as e:
        state["error"] = f"4コマ画像生成エラー: {e}"

    return state


def generate_subtitle_images(
    state: Dict[str, Any], attempt_prefecture_image: bool = True
) -> Dict[str, Any]:
    """サブタイトル画像を生成"""
    if not attempt_prefecture_image:
        state["subtitle_image_paths"] = None
        return state

    try:
        settings = get_env_config()
        gcp_project_id = settings.get("gcp_project_id")
        gcp_location = settings.get("gcp_location")

        if not all([gcp_project_id, gcp_location]):
            raise ValueError("GCP Project ID or Location not configured.")

        image_paths = generate_prefecture_image_and_get_path(
            prefecture=state["selected_prefecture_name"],
            main_title=state["main_title"],
            sub_titles=state["subtitles"],
            gcp_project_id=gcp_project_id,
            gcp_location=gcp_location,
            llm_model_name=settings.get("model_name", "gemini-1.5-pro-001"),
            image_gen_model_name=settings.get(
                "image_gen_model_name", "imagen-3.0-fast-generate-001"
            ),
        )
        state["subtitle_image_paths"] = image_paths

    except Exception as e:
        state["error"] = f"サブ画像生成エラー: {e}"

    return state


def format_html(state: Dict[str, Any]) -> Dict[str, Any]:
    """HTMLを整形"""
    article_title = state.get("initial_article_title") or state.get(
        "main_title", "生成記事"
    )
    subtitles = state.get("subtitles", [])
    blocks = state.get("generated_article_json", {}).get("block", [])
    main_img = state.get("main_theme_image_path")
    sub_imgs = state.get("subtitle_image_paths", [])
    aphorism = state.get("aphorism")
    error = state.get("error")

    final_html = build_html_article(
        article_title=article_title,
        subtitles=subtitles,
        blocks=blocks,
        main_img=main_img,
        sub_imgs=sub_imgs,
        aphorism=aphorism,
        error=error,
    )

    state["html_output"] = final_html
    return state
