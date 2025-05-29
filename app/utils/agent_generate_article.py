import os
import json
from typing import TypedDict, List, Dict, Any
import traceback

from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field, validator

# (prompts.GENERATE_ARTICLE は直接文字列として定義するため、ここでは不要とします)
# from prompts.GENERATE_ARTICLE import GENERATE_ARTICLE_PROMPT
from config.env_config import get_env_config

from langchain_google_community.search import GoogleSearchAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langgraph.graph import StateGraph, END

# --- Pydanticモデル定義 (新しい記事構造) ---
class ArticleSection(BaseModel):
    subtitle: str = Field(description="セクションのサブタイトル")
    content: str = Field(description="セクションの本文。段落間は改行2つ(\\n\\n)で区切る。")

class StructuredArticle(BaseModel):
    main_title: str = Field(description="記事のメインタイトル")
    sections: List[ArticleSection] = Field(description="記事のセクションリスト。必ず5つのセクションを持つこと。")

    @validator('sections')
    def check_number_of_sections(cls, v):
        if len(v) != 5:
            # LLMの出力が厳密に5つにならない場合も考慮し、ここでは警告に留めるか、
            # あるいはパースエラーとするかは要件による。
            # ここではPydanticのバリデーションとしてはエラーとする。
            # 実際の運用ではLLMへのプロンプトで強く制約する。
            # raise ValueError('記事は厳密に5つのセクションを持つ必要があります。')
            print(f"警告: セクションの数が5つではありませんでした (実際の数: {len(v)})。プロンプトでの制御を推奨します。")
        return v

# --- 状態の定義 ---
class AgentState(TypedDict):
    topic: str
    search_query: str
    raw_search_results: List[Dict[str, Any]]
    scraped_context: str
    generated_article_json: Dict[str, Any]  # StructuredArticle.model_dump() の結果
    initial_main_title: str # ★ MODIFIED: メインタイトル用
    initial_article_content: str # 全セクションの本文を結合したもの
    revised_article_json_str: str | None # 添削後の記事全体(StructuredArticle形式)のJSON文字列
    html_output: str
    error: str | None

# ★ MODIFIED: プロンプトを直接定義
GENERATE_STRUCTURED_ARTICLE_PROMPT_TEMPLATE = """あなたはプロのWEBライターです。
提供されたトピックと検索結果(ウェブページからの抜粋情報)に基づいて、読者にとって魅力的で、正確な情報に基づいた記事を生成してください。
記事は以下のJSON形式に従って、1つのメインタイトルと、厳密に5つのセクションで構成してください。
各セクションにはサブタイトルと、そのサブタイトルに沿った内容の本文が必要です。
本文の各段落は改行2つ（\\n\\n）で区切ってください。

トピック: {topic}

検索結果からの抜粋情報:
{search_results}

{format_instructions}
"""


def generate_article_workflow(
    topic_input: str,
) -> Dict[str, Any]:
    print(f"\n--- 「{topic_input}」に関する記事生成を開始します (新構造版) ---")

    settings = get_env_config()
    google_api_key = settings.get("google_api_key")
    google_cse_id = settings.get("google_cse_id")

    if not google_api_key or not google_cse_id:
        error_msg = "エラー: Google APIキーまたはCSE IDが設定ファイルに存在しません。"
        # (中略: エラーリターン部分は変更なし)
        print(error_msg)
        return {
            "success": False,
            "topic": topic_input,
            "html_output": None,
            "error_message": error_msg,
            "final_state_summary": None,
        }

    try:
        llm = ChatVertexAI(
            model_name=settings.get("model_name", "gemini-1.0-pro-001"), # gemini-1.5-pro-preview-0409 など、より長いコンテキストに対応できるモデルを推奨
            temperature=0.1, # 少し創造性を加えるために若干上げることを検討
            max_output_tokens=settings.get("max_output_tokens", 8192),
            max_retries=6,
            stop=None,
        )
        search_wrapper = GoogleSearchAPIWrapper(
            google_api_key=google_api_key,
            google_cse_id=google_cse_id,
        )
        # ★ MODIFIED: 新しいPydanticモデルでパーサーを初期化
        structured_article_parser = PydanticOutputParser(pydantic_object=StructuredArticle)
    except Exception as e_init:
        # (中略: エラーリターン部分は変更なし)
        error_msg = f"LLMまたはツールの初期化中にエラーが発生しました: {e_init}"
        print(error_msg)
        traceback.print_exc()
        return {
            "success": False,
            "topic": topic_input,
            "html_output": None,
            "error_message": error_msg,
            "final_state_summary": None,
        }

    # --- ノード定義 ---
    # generate_search_query_node, Google Search_node, scrape_and_prepare_context_node は変更なし
    def generate_search_query_node(state: AgentState) -> AgentState:
        print("--- ステップ1a: 検索クエリ生成 ---")
        topic = state["topic"]
        # クエリを少し具体的にして、多様な情報が得られるように調整も可能
        search_query = f"{topic} 詳細解説 メリット デメリット ポイント 最新情報"
        print(f"生成された検索クエリ: {search_query}")
        return {**state, "search_query": search_query, "error": None}

    def google_search_node(state: AgentState) -> AgentState:
        if state.get("error"): return state
        print("--- ステップ1b: Google検索実行 ---")
        query = state["search_query"]
        num_results = settings.get("num_search_results", 5) # 情報を増やすために検索結果数を検討
        try:
            print(f"Google検索中 (クエリ: {query}, {num_results}件)...")
            search_results_list = search_wrapper.results(query=query, num_results=num_results)
            print(f"検索結果 {len(search_results_list) if search_results_list else 0} 件取得完了。")
            return {**state, "raw_search_results": (search_results_list if search_results_list else []), "error": None }
        except Exception as e:
            print(f"Google検索中にエラーが発生しました: {e}")
            return {**state, "error": f"Google検索エラー: {str(e)}"}

    def scrape_and_prepare_context_node(state: AgentState) -> AgentState:
        if state.get("error"): return state
        print("--- ステップ1c: Webスクレイピングとコンテキスト準備 ---")
        search_results_list = state["raw_search_results"]
        scraped_contents = []
        # コンテキスト長を増やすために、1ページあたりの最大文字数を増やすことを検討 (LLMの最大入力トークン数と相談)
        max_content_length_per_page = settings.get("max_content_length_per_page_scrape", 2500)
        # (スクレイピングロジックの主要部分は変更なし、詳細なログ出力は維持)
        print(f"受け取った検索結果の数: {len(search_results_list)}")
        if search_results_list: print(f"最初の検索結果のサンプル: {search_results_list[0]}")
        if not search_results_list:
            print("検索結果が空のため、スクレイピングをスキップします。")
            return {**state, "scraped_context": "関連情報は見つかりませんでした。", "error": None}
        print(f"検索結果から上位{len(search_results_list)}件のウェブページを読み込んでいます...")
        for i, result in enumerate(search_results_list):
            title = result.get("title", "タイトルなし")
            link = result.get("link")
            snippet = result.get("snippet", "スニペットなし")
            if link:
                try:
                    print(f"  読み込み中 ({i+1}/{len(search_results_list)}): {link}")
                    loader = WebBaseLoader(web_path=link, requests_kwargs={"timeout": 20, "headers": {"User-Agent": "..."}}) # UserAgentは省略
                    documents = loader.load()
                    if documents:
                        bs_transformer = BeautifulSoupTransformer()
                        docs_transformed = bs_transformer.transform_documents(documents, tags_to_extract=["div", "p", "h1", "h2", "h3", "li", "article", "main", "section"])
                        page_content_after_transform = " ".join([doc.page_content for doc in docs_transformed])
                        if page_content_after_transform:
                            shortened_content = page_content_after_transform[:max_content_length_per_page].strip()
                            if shortened_content:
                                scraped_contents.append(f"参照元URL: {link}\nタイトル: {title}\n内容:\n{shortened_content}")
                            else: scraped_contents.append(f"参照元URL: {link}\nタイトル: {title}\n概要: {snippet}\n(主要コンテンツ抽出失敗)")
                        else: scraped_contents.append(f"参照元URL: {link}\nタイトル: {title}\n概要: {snippet}\n(BS適用後空)")
                    else: scraped_contents.append(f"参照元URL: {link}\nタイトル: {title}\n概要: {snippet}\n(WebBaseLoaderドキュメントなし)")
                except Exception as e_scrape:
                    traceback.print_exc()
                    scraped_contents.append(f"参照元URL: {link}\nタイトル: {title}\n概要: {snippet}\n(エラー: {type(e_scrape).__name__})")
            else: scraped_contents.append(f"タイトル: {title}\n概要: {snippet} (URLなし)")
        search_context_str = "\n\n===\n\n".join(scraped_contents) if scraped_contents else "関連性の高いウェブページのコンテンツは見つかりませんでした。"
        print(f"ウェブページ読み込み完了。最終的なscraped_contextの先頭200文字: {search_context_str[:200]}")
        return {**state, "scraped_context": search_context_str, "error": None}


    def generate_structured_article_node(state: AgentState) -> AgentState: # ★ MODIFIED
        if state.get("error"):
            return state
        print("--- ステップ2: 構造化記事生成 (新構造) ---")
        topic = state["topic"]
        search_context = state["scraped_context"]
        try:
            system_message = "あなたはプロのWEBライターです。" # 実際のプロンプトはテンプレートで詳細に指示
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_message), # System メッセージは簡潔に役割を
                    ("user", GENERATE_STRUCTURED_ARTICLE_PROMPT_TEMPLATE), # 詳細指示はユーザープロンプトで
                ]
            ).partial(format_instructions=structured_article_parser.get_format_instructions())
            
            chain = prompt | llm | structured_article_parser
            print("LLMによる記事生成中 (新構造)...")
            input_data = {"topic": topic, "search_results": search_context}
            
            parsed_article_obj: StructuredArticle = chain.invoke(input_data)
            
            generated_article_json = parsed_article_obj.model_dump()
            main_title = parsed_article_obj.main_title
            
            # initial_article_content に全セクションの本文を結合
            all_sections_content = "\n\n".join(
                [section.content for section in parsed_article_obj.sections]
            )
            
            print(f"記事生成完了。メインタイトル: {main_title}")
            if len(parsed_article_obj.sections) != 5:
                print(f"警告：生成されたセクション数が5ではありません: {len(parsed_article_obj.sections)}。LLMへの指示を確認してください。")


            return {
                **state,
                "generated_article_json": generated_article_json,
                "initial_main_title": main_title,
                "initial_article_content": all_sections_content,
                "error": None,
            }
        except OutputParserException as e_parse:
            llm_output_str = getattr(e_parse, "llm_output", str(e_parse.args[0] if e_parse.args else str(e_parse)))
            error_msg = f"記事生成中にOutputParserエラー: {e_parse}\nLLM Raw Output:\n{llm_output_str}"
            print(error_msg)
            return {**state, "error": error_msg}
        except Exception as e:
            print(f"記事生成中に予期せぬエラー: {e}")
            traceback.print_exc()
            return {**state, "error": f"記事生成エラー: {str(e)}"}

    def revise_article_node(state: AgentState) -> AgentState: # ★ MODIFIED
        if state.get("error"):
            return state
        print("--- ステップ3: 記事添削 (新構造JSON形式で出力) ---")
        
        # generated_article_json (辞書) を添削のベースとする
        article_to_revise_dict = state.get("generated_article_json")

        if not article_to_revise_dict or not article_to_revise_dict.get("sections"):
            print("添削対象の記事データ(JSON)が不完全、またはありません。スキップします。")
            # スキップする場合、元のJSON文字列をrevised_article_json_strに入れておく
            # エラーがない場合はそのまま、エラーがある場合はエラーメッセージを付与
            error_msg_skip = state.get("error")
            if not error_msg_skip:
                error_msg_skip = "添削対象記事なしのため、添削スキップ。"

            return {
                **state,
                "revised_article_json_str": json.dumps(article_to_revise_dict) if article_to_revise_dict else None,
                "error": error_msg_skip
            }
        
        try:
            # 辞書をJSON文字列に変換してプロンプトに含める
            article_to_revise_json_str = json.dumps(article_to_revise_dict, ensure_ascii=False, indent=2)
            print("記事を添削し、メインタイトルと5セクション(サブタイトル+本文)をJSONで取得します...")

            prompt_text = f"""以下のJSON形式で提供される記事を、より自然で読みやすく、誤字脱字や文法的な誤りがないように添削してください。
記事の主要な内容と構造（メインタイトル1つ、セクション5つ、各セクションにサブタイトルと本文）は変えずに、表現を改善してください。
ですます調を維持してください。
各セクションの本文の段落は改行2つ（\\n\\n）で区切ってください。

元の記事 (JSON形式):
{article_to_revise_json_str}

添削後の記事 (同じJSON形式で、メインタイトル、各セクションのサブタイトルと本文を添削して返してください):
"""
            response = llm.invoke([HumanMessage(content=prompt_text)])
            revised_json_str_output = response.content
            print(f"記事添削完了 (LLMからのJSON文字列):\n{revised_json_str_output[:400]}...")

            # JSONバリデーション (新しい構造に合わせて)
            try:
                parsed_json = json.loads(revised_json_str_output)
                # StructuredArticle モデルでバリデーションを試みる (任意)
                # validated_article = StructuredArticle(**parsed_json) # これで形式もチェック
                # 簡単なチェック:
                if not isinstance(parsed_json, dict) or \
                "main_title" not in parsed_json or \
                "sections" not in parsed_json or \
                not isinstance(parsed_json["sections"], list) or \
                len(parsed_json["sections"]) != 5: # 厳密に5セクションを期待
                    num_sections = len(parsed_json.get("sections", [])) if isinstance(parsed_json.get("sections"), list) else 0
                    warning_msg = f"添削結果のJSON構造が期待通りではありません (main_title, sections(5件) が必要)。実際のセクション数: {num_sections}"
                    print(warning_msg)
                    # エラーとするか、そのまま使うかは運用次第。ここでは警告に留め、元のものを一部使うなどフォールバックを検討。
                    # 今回は、構造が大きく崩れていなければそのまま通し、HTML側で不足分を処理する。
                    # ただし、深刻なパースエラーはエラー扱い。
                    if num_sections != 5:
                        # 致命的ではないが、ログには残す
                        current_error = state.get("error")
                        state_error_msg = f"{current_error}\n{warning_msg}" if current_error else warning_msg
                        return {**state, "revised_article_json_str": revised_json_str_output, "error": state_error_msg}


                print(f"添削JSONパース成功。メインタイトル: {parsed_json.get('main_title')}")
                return {**state, "revised_article_json_str": revised_json_str_output, "error": state.get("error")} # 元のエラーを引き継ぐ
            except json.JSONDecodeError as e_json:
                error_msg = f"添削結果のJSONパースに失敗: {e_json}. LLM Output: {revised_json_str_output}"
                print(error_msg)
                current_error = state.get("error")
                final_error = f"{current_error}\n{error_msg}" if current_error else error_msg
                # フォールバックとして添削前のJSON文字列を使用
                return {**state, "revised_article_json_str": article_to_revise_json_str, "error": final_error}
            # except Exception as e_val: # Pydanticバリデーションを使う場合
            #     error_msg = f"添削結果のJSON構造が不正(Pydantic): {e_val}. LLM Output: {revised_json_str_output}"
            #     print(error_msg)
            #     # (上記と同様のフォールバック処理)
            #     return {**state, "revised_article_json_str": article_to_revise_json_str, "error": f"{state.get('error')}\n{error_msg}" if state.get('error') else error_msg}


        except Exception as e:
            print(f"記事添削中にエラー: {e}")
            traceback.print_exc()
            current_error = state.get("error")
            error_msg_revise = f"記事添削エラー: {str(e)}"
            final_error = f"{current_error}\n{error_msg_revise}" if current_error else error_msg_revise
            return {**state, "revised_article_json_str": json.dumps(article_to_revise_dict) if article_to_revise_dict else None, "error": final_error}


    def format_html_node(state: AgentState) -> AgentState: # ★ MODIFIED
        print("--- ステップ5: HTML整形 (新構造) ---")

        html_main_title = state.get("topic", "記事") # デフォルト
        article_html_sections = ""
        current_error_msg = state.get("error")

        data_source_for_html = None
        revised_json_str = state.get("revised_article_json_str")

        if revised_json_str:
            try:
                data_source_for_html = json.loads(revised_json_str)
                print("添削済みJSONデータをHTML生成に使用します。")
            except json.JSONDecodeError as e:
                print(f"添削済みJSONのパースに失敗: {e}。初期生成記事データでフォールバックします。")
                error_parse_html = f"HTML生成のための添削JSONパース失敗: {e}"
                current_error_msg = f"{current_error_msg}\n{error_parse_html}" if current_error_msg else error_parse_html
                data_source_for_html = state.get("generated_article_json") # これは既に辞書型
                if data_source_for_html:
                    print("初期生成記事データ(JSON辞書)をHTML生成に使用します。")
        elif not current_error_msg: # revised_json_str がなく、他にエラーもない場合
            data_source_for_html = state.get("generated_article_json")
            if data_source_for_html:
                print("初期生成記事データ(JSON辞書)をHTML生成に使用します。")
        
        if data_source_for_html and isinstance(data_source_for_html, dict):
            html_main_title = data_source_for_html.get("main_title", html_main_title)
            sections_data = data_source_for_html.get("sections", [])
            
            html_sections_parts = []
            if isinstance(sections_data, list):
                for i, section_item in enumerate(sections_data):
                    if isinstance(section_item, dict):
                        subtitle = section_item.get("subtitle", f"サブタイトル {i+1} (不明)")
                        content = section_item.get("content", "このセクションの本文はありません。")
                        
                        subtitle_html = f"<h2>{subtitle}</h2>\n"
                        # content が "\n\n" で区切られた文字列と仮定
                        paragraphs = content.strip().split("\n\n")
                        content_html = "".join([f"<p>{p.strip()}</p>\n" for p in paragraphs if p.strip()])
                        html_sections_parts.append(subtitle_html + content_html)
                    else: # セクションアイテムが辞書でない場合
                        html_sections_parts.append(f"<h2>セクション {i+1} (データ不正)</h2><p>このセクションのデータ形式が正しくありません。</p>")
                article_html_sections = "\n".join(html_sections_parts)
            else: # sectionsデータがリストでない場合
                article_html_sections = "<p>記事のセクションデータを正しく読み込めませんでした。</p>"
        
        if current_error_msg and not article_html_sections:
            article_html_sections = f"<h1>記事コンテンツ生成エラー</h1><p>エラーのため記事の主要部分を生成できませんでした。</p><p>エラー詳細: <pre>{current_error_msg}</pre></p>"
            if not html_main_title or html_main_title == state.get("topic", "記事"):
                html_main_title = "記事生成エラー"
        elif not article_html_sections and not current_error_msg :
            article_html_sections = "<p>記事が生成されませんでした。</p>"


        # HTMLテンプレート (CSSは前回と同様のため省略)
        html_output_content = f"""<!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{html_main_title}</title>
            <style>
                body {{ font-family: 'Georgia', serif; line-height: 1.7; margin: 0; padding: 0; background-color: #f4f1ea; color: #3a3a3a; }}
                .container {{ max-width: 750px; margin: 50px auto; background-color: #fffdf7; padding: 40px 50px; border-radius: 4px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); border-left: 6px solid #a0522d; }}
                h1 {{ font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 2.4em; color: #4a3b32; border-bottom: 1px solid #dcdcdc; padding-bottom: 20px; margin-top: 0; margin-bottom: 35px; font-weight: bold; letter-spacing: 0.5px; }}
                h2 {{ font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 1.8em; color: #5a473a; margin-top: 40px; margin-bottom: 20px; border-bottom: 1px dashed #e0e0e0; padding-bottom: 10px; }}
                p {{ margin-bottom: 1.8em; font-size: 1.1em; color: #484848; text-align: justify; orphans: 3; widows: 3; }}
                /* (blockquote, article-footer スタイルは省略。前回と同様) */
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{html_main_title}</h1>
                {article_html_sections}
            </div>
        </body>
        </html>"""
        print("HTML整形完了 (新構造)。")
        return {
            **state,
            "html_output": html_output_content,
            "error": current_error_msg, # 途中で発生したエラーも保持
        }

    def error_handler_node(state: AgentState) -> AgentState:
        print(f"--- エラー発生 (エラーハンドラノード) ---")
        error_message = state.get("error", "不明なエラー")
        print(f"エラー内容: {error_message}")
        # エラー時のHTML出力 (主要部分は変更なし)
        html_error_output_content = f"""<!DOCTYPE html><html lang="ja">... (エラー表示HTML) ...</html>""" # 省略
        # (中略: エラーハンドラノードのHTML生成とリターンは変更なし)
        html_error_output_content = f"""<!DOCTYPE html><html lang="ja"><head><meta charset="UTF-8"><title>処理エラー</title></head><body><h1>記事生成プロセスでエラーが発生しました</h1><p><strong>エラーメッセージ:</strong></p><pre>{error_message}</pre><hr><p><strong>状態情報 (一部):</strong></p><pre>トピック: {state.get("topic")}\nメインタイトル(初期): {state.get("initial_main_title")}\n添削記事JSONの有無: {"あり" if state.get("revised_article_json_str") else "なし"}</pre></body></html>"""

        return {
            **state,
            "html_output": html_error_output_content,
        }

    # --- グラフ構築 (ノード名変更なし、接続も基本変更なし) ---
    workflow = StateGraph(AgentState)
    # (add_node, set_entry_point, add_conditional_edges, add_edge の部分は変更なし)
    workflow.add_node("generate_search_query", generate_search_query_node)
    workflow.add_node("google_search", google_search_node)
    workflow.add_node("scrape_and_prepare_context", scrape_and_prepare_context_node)
    workflow.add_node("generate_structured_article", generate_structured_article_node) # 内容変更あり
    workflow.add_node("revise_article", revise_article_node) # 内容変更あり
    workflow.add_node("format_html", format_html_node) # 内容変更あり
    workflow.add_node("error_handler", error_handler_node)

    workflow.set_entry_point("generate_search_query")

    def should_continue_or_handle_error(state: AgentState) -> str:
        if state.get("error"):
            print(f"エラー検出: {state.get('error')[:100]}... error_handlerへ遷移します。") # type: ignore
            return "error_handler"
        print("処理継続: 次のノードへ進みます。")
        return "continue"

    workflow.add_conditional_edges("generate_search_query", should_continue_or_handle_error, {"continue": "google_search", "error_handler": "error_handler"})
    workflow.add_conditional_edges("google_search", should_continue_or_handle_error, {"continue": "scrape_and_prepare_context", "error_handler": "error_handler"})
    workflow.add_conditional_edges("scrape_and_prepare_context", should_continue_or_handle_error, {"continue": "generate_structured_article", "error_handler": "error_handler"})
    workflow.add_conditional_edges("generate_structured_article", should_continue_or_handle_error, {"continue": "revise_article", "error_handler": "error_handler"})
    workflow.add_conditional_edges("revise_article", should_continue_or_handle_error, {"continue": "format_html", "error_handler": "error_handler"})
    workflow.add_edge("format_html", END)
    workflow.add_edge("error_handler", END)


    try:
        app = workflow.compile()
    except Exception as e_compile:
        # (中略: エラーリターン部分は変更なし)
        error_msg = f"ワークフローグラフのコンパイル中にエラー: {e_compile}"
        print(error_msg); traceback.print_exc()
        return {"success": False, "topic": topic_input, "html_output": None, "error_message": error_msg, "final_state_summary": None}


    initial_state: AgentState = { 
        "topic": topic_input,
        "search_query": "",
        "raw_search_results": [],
        "scraped_context": "",
        "generated_article_json": {}, # 初期は空の辞書
        "initial_main_title": "",    # 初期は空
        "initial_article_content": "", # 初期は空
        "revised_article_json_str": None, # 初期はNone
        "html_output": "",
        "error": None,
    }

    final_state = None
    try:
        for event_part in app.stream(initial_state, {"recursion_limit": 15}): # recursion_limit は必要に応じて調整
            for node_name, current_state_after_node in event_part.items():
                print(f"\n[ノード完了] '{node_name}'")
                print(f"  エラー状態: {current_state_after_node.get('error')}")
                final_state = current_state_after_node
                if node_name == "__end__": print("  ワークフロー終了点に到達。")
    except Exception as e_stream:
        error_msg = f"ワークフロー実行中にエラー: {e_stream}"; print(error_msg); traceback.print_exc()
        current_html_output = None; current_error_message_from_state = None
        summary_dict_content = {"topic": topic_input, "error": error_msg}
        if isinstance(final_state, dict):
            current_html_output = final_state.get("html_output")
            current_error_message_from_state = final_state.get("error")
            summary_dict_content = {k: v for k, v in final_state.items() if k != "raw_search_results"}
            if "error" not in summary_dict_content : summary_dict_content["error"] = error_msg
            elif current_error_message_from_state and error_msg not in current_error_message_from_state:
                summary_dict_content["error"] = f"{current_error_message_from_state}\nSTREAM_ERROR: {error_msg}"
        final_error_message_for_return = current_error_message_from_state or error_msg
        if current_error_message_from_state and error_msg not in current_error_message_from_state:
            final_error_message_for_return = f"{current_error_message_from_state}\nSTREAM_ERROR: {error_msg}"
        return {"success": False, "topic": topic_input, "html_output": current_html_output, "error_message": final_error_message_for_return, "final_state_summary": summary_dict_content}

    print("\n--- 全ての処理が完了しました ---")
    if final_state:
        error_message_from_state = final_state.get("error")
        success_status = not bool(error_message_from_state)
        html_content_output = final_state.get("html_output", "")
        final_state_summary_dict = {k: v for k, v in final_state.items() if k != "raw_search_results"}
        if "html_output" not in final_state_summary_dict: final_state_summary_dict["html_output"] = html_content_output
        print(f"HTML Outputの先頭100文字: {html_content_output[:100] if html_content_output else '（HTMLなし）'}")
        return {"success": success_status, "topic": topic_input, "html_output": html_content_output, "error_message": error_message_from_state, "final_state_summary": final_state_summary_dict}
    else:
        error_msg_no_final_state = "最終状態が取得できませんでした (ワークフロー実行で予期せぬ問題)。"
        print(error_msg_no_final_state)
        error_html_for_no_final_state = f"<!DOCTYPE html><html lang='ja'><head><title>重大なエラー</title></head><body><h1>ワークフロー処理で重大なエラー</h1><p>{error_msg_no_final_state}</p></body></html>"
        return {"success": False, "topic": topic_input, "html_output": error_html_for_no_final_state, "error_message": error_msg_no_final_state, "final_state_summary": None}