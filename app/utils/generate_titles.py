import streamlit as st
from config.env_config import get_env_config
from langchain_google_vertexai import ChatVertexAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain_core.exceptions import OutputParserException
import traceback
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any, Union
from pydantic import BaseModel, Field
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
)  # 必要なものをインポート
from langchain_core.messages import AIMessage  # LLM出力の型ヒント用
from langchain_google_community.search import GoogleSearchAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from prompts.PHILOSOPHICAL_TITLES_PROMPT import PHILOSOPHICAL_TITLES_PROMPT


# --- Pydanticモデル定義 (main_titleのdescriptionを修正) ---
class TitlesOutput(BaseModel):
    main_title: str = Field(
        description="生成されたメインタイトル。必ず20文字以上30文字以内で、指定された都道府県名を含むこと。" 
    )
    sub_titles: List[str] = Field(
        description="生成されたサブタイトルのリスト。5つのサブタイトルを含むこと。",
        min_items=5,
        max_items=5,
    )


def _get_search_results(
    query: str, api_key: str, cse_id: str, num_results: int
) -> List[Dict[str, Any]]:
    """Google検索を実行し、結果のリストを返す。"""
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
    return search_results_list if search_results_list else []


def _scrape_and_prepare_context(
    search_results_list: List[Dict[str, Any]], settings: dict
) -> str:
    """検索結果のURLからウェブページをスクレイピングし、LLM用コンテキスト文字列を作成する。"""
    scraped_contents = []
    if not search_results_list:
        return "関連情報は見つかりませんでした。"

    max_content_length_per_page = settings.get("max_content_length_per_page", 2000)

    for i, result in enumerate(search_results_list):
        title = result.get("title", "タイトルなし")
        link = result.get("link")
        snippet = result.get("snippet", "スニペットなし")

        if link:
            try:
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

    return search_context_str


def _invoke_llm_for_titles(
    selected_prefecture: str, search_context: str, settings: dict
) -> Union[TitlesOutput, dict]:
    """LLMチェーンを準備・実行し、パースされたタイトル群またはエラー情報を返す。"""
    llm = ChatVertexAI(
        model_name=settings.get("model_name", "gemini-1.0-pro-001"),
        temperature=0,
        max_output_tokens=settings.get("max_output_tokens", 2048),
        max_retries=settings.get("llm_max_retries", 6),
        stop=None,
    )

    pydantic_parser = PydanticOutputParser(pydantic_object=TitlesOutput)

    system_template = (
        "あなたは、ユーザーから与えられた指示とフォーマットに厳密に従って、"
        "指定されたJSON形式で応答を生成するAIアシスタントです。"
        "応答には指示されたJSON以外の文字列（説明、前置き、後書き、マークダウンなど）を一切含めないでください。"
    )

    prompt_template_obj = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            ("user", PHILOSOPHICAL_TITLES_PROMPT),
        ]
    ).partial(format_instructions=pydantic_parser.get_format_instructions())

    # 1. RetryWithErrorOutputParser インスタンスを作成
    #    llm とラップするパーサー (pydantic_parser) を渡します。
    retry_parser = RetryWithErrorOutputParser.from_llm(
        llm=llm,
        parser=pydantic_parser,  # LLMに修正を促す際に、このパーサーの形式指示が再度使われる
        max_retries=settings.get("parser_max_retries", 3),
    )

    # 2. LCELチェーンの構築
    #    parse_with_prompt を呼び出すために、PromptValue と LLM出力文字列を準備する

    # ヘルパー関数 (RunnableLambda内で使用)
    def extract_prompt_value(inputs: Dict):
        return prompt_template_obj.invoke(inputs)

    def get_llm_completion_str(inputs: Dict) -> str:
        # (prompt | llm) の結果 (AIMessage) から content (文字列) を取り出す
        ai_message: AIMessage = (prompt_template_obj | llm).invoke(inputs)
        return ai_message.content

    # チェーンの定義
    # RunnablePassthrough.assign を使って、入力辞書に新しいキー (prompt_value, completion) を追加する
    # これらの新しいキーの値は、それぞれ指定されたRunnableを実行して得られる
    chain_with_intermediate_results = RunnablePassthrough.assign(
        prompt_value=RunnableLambda(
            extract_prompt_value
        ),  # 入力辞書 x を extract_prompt_value に渡す
        completion=RunnableLambda(
            get_llm_completion_str
        ),  # 入力辞書 x を get_llm_completion_str に渡す
    )

    # 上記の結果 (prompt_value と completion を含む辞書) を使って、
    # retry_parser の parse_with_prompt を呼び出す
    final_chain = chain_with_intermediate_results | RunnableLambda(
        lambda x: retry_parser.parse_with_prompt(
            completion=x["completion"], prompt_value=x["prompt_value"]
        )
    )

    input_data = {
        "selected_prefecture": selected_prefecture,
        "search_results": search_context,
    }

    try:
        parsed_titles: TitlesOutput = final_chain.invoke(input_data)
        return parsed_titles

    except OutputParserException as e:
        llm_output = getattr(e, "llm_output", str(e.args[0] if e.args else str(e)))
        error_message = f"⚠️ Output Parser Error (after retries): {e}\n"
        error_message += f"LLMからの生の応答がJSON形式ではありませんでした。リトライ処理も失敗しました。\n"
        if llm_output:
            error_message += (
                f"LLM Raw Response (during parsing attempt):\n---\n{llm_output}\n---"
            )

        print(error_message)
        st.error(error_message)
        if llm_output:
            st.text_area(
                "LLM Raw Output (from parser exception):", llm_output, height=200
            )

        return {
            "error": "Output Parser Error",
            "raw_response": llm_output,
            "details": str(e),
        }
    except Exception as e:
        error_message = (
            f"LLM呼び出しまたはチェーン実行中に予期せぬエラーが発生しました: {e}"
        )
        print(error_message)
        traceback.print_exc()
        st.error(error_message)
        if hasattr(e, "args") and e.args:
            st.text_area("Error Details:", str(e.args[0]), height=100)
        return {
            "error": "LLM Invocation or Chain Execution Error",
            "details": str(e),
        }


def generate_titles_for_prefecture(selected_prefecture: str) -> dict:
    settings = get_env_config()

    google_api_key = settings.get("google_api_key")
    google_cse_id = settings.get("google_cse_id")

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
        st.error(f"LLMからの応答が予期せぬ形式でした: {type(llm_response_or_titles)}")
        if isinstance(
            llm_response_or_titles, str
        ):  # LLMの生の文字列がそのまま返ってきた場合など
            st.text_area(
                "Unexpected LLM Response (Raw String):",
                llm_response_or_titles,
                height=100,
            )

        return {
            "error": "Unexpected LLM Response",
            "details": f"The response from the LLM was not in the expected format. Type: {type(llm_response_or_titles)}",
            "search_results_for_display": raw_search_results_for_display,
            "titles_output": None,
        }
