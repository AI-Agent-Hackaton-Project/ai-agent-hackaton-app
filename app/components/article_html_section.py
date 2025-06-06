import streamlit as st

from utils.agent_generate_article import generate_article_workflow
from utils.generate_titles import generate_titles_for_prefecture


def initialize_session_state():
    """
    アプリケーションのセッションステート変数を初期化します。
    これらの変数は、タイトルや記事の生成状態を管理するために使用されます。
    """
    if "main_title_generated" not in st.session_state:
        st.session_state.main_title_generated = None
    if "sub_titles_generated" not in st.session_state:
        st.session_state.sub_titles_generated = None
    if "titles_generated_successfully" not in st.session_state:
        st.session_state.titles_generated_successfully = False


def _execute_and_display_article_generation(
    main_title_for_article, sub_titles_for_article
):
    """
    指定されたタイトルに基づいて記事を生成し、結果をStreamlit UIに表示します。
    """
    if not main_title_for_article or not sub_titles_for_article:
        st.error(
            "記事を生成するためのタイトル情報が見つかりません。再度タイトルを生成してください。"
        )
        return

    with st.spinner(f"「{main_title_for_article}」の記事を生成中です..."):
        article_result = generate_article_workflow(
            main_title_for_article, sub_titles_for_article
        )

    st.subheader("📄 記事生成結果")

    displayed_topic = article_result.get("main_title", main_title_for_article)

    if article_result.get("success"):
        article_content_preview = article_result.get("html_output")
        if article_content_preview:
            st.html(article_content_preview)
        else:
            st.warning("生成された記事のHTMLコンテンツがありません。")
    else:
        st.error(f"❌ 記事「{displayed_topic}」の生成に失敗しました。")
        if article_result.get("error_message"):
            st.error(f"**エラーメッセージ:** {article_result.get('error_message')}")
        error_html_output = article_result.get("html_output")
        if error_html_output:  # エラー時でもHTML出力があれば表示
            st.markdown("#### エラー時のHTML出力プレビュー:")
            st.html(error_html_output)


def render_title_generation_section(selected_prefecture_name):
    """
    タイトル生成のためのUIコンポーネントを描画し、関連するロジックを実行します。
    ユーザーが都道府県を選択し、タイトル生成ボタンを押すと、APIを呼び出し結果を表示します。
    成功した場合、生成されたタイトルはセッションステートに保存され、続けて記事生成が自動的に開始されます。

    Args:
        selected_prefecture_name (str): 選択された都道府県名。
    """

    if st.button(
        f"{selected_prefecture_name}のタイトルと記事を生成する",
        key="generate_titles_and_article_button",
    ):
        st.session_state.titles_generated_successfully = False
        st.session_state.main_title_generated = None
        st.session_state.sub_titles_generated = None

        with st.spinner(
            f"{selected_prefecture_name}の情報を検索し、タイトルを生成しています..."
        ):
            result_titles = generate_titles_for_prefecture(selected_prefecture_name)

        if result_titles.get("error"):
            st.error(f"エラーが発生しました: {result_titles['error']}")
            st.json(result_titles.get("details", "詳細不明"))
            if "raw_response" in result_titles:
                st.text_area(
                    "LLM Raw Response (on error):",
                    result_titles["raw_response"],
                    height=200,
                )
            st.session_state.titles_generated_successfully = False
        elif result_titles.get("titles_output"):
            generated_main_title = result_titles["titles_output"]["main_title"]
            generated_sub_titles = result_titles["titles_output"]["sub_titles"]

            st.session_state.main_title_generated = generated_main_title
            st.session_state.sub_titles_generated = generated_sub_titles
            st.session_state.titles_generated_successfully = True

            _execute_and_display_article_generation(
                generated_main_title, generated_sub_titles
            )

        else:
            st.warning(
                "タイトルを生成できませんでした。APIからの応答に必要な情報が含まれていません。"
            )
            st.session_state.titles_generated_successfully = False


def article_generator_app(selected_prefecture_name):
    """
    AI記事ジェネレーターアプリケーションのメイン関数。
    ページ設定、セッションステート初期化、タイトル生成とそれに続く記事生成のUIとロジックを実行します。
    """
    initialize_session_state()
    render_title_generation_section(selected_prefecture_name)
