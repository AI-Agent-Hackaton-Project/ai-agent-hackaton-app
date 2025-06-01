import streamlit as st
import os  # 元のコードに含まれていたため維持

# 外部の関数や定数を想定 (実際の利用時は適切な場所からインポートしてください)
from utils.agent_generate_article import generate_article_workflow
from utils.generate_titles import generate_titles_for_prefecture
from config.constants import JAPAN_PREFECTURES


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


def render_title_generation_section(selected_prefecture_name):
    """
    タイトル生成のためのUIコンポーネントを描画し、関連するロジックを実行します。
    ユーザーが都道府県を選択し、タイトル生成ボタンを押すと、APIを呼び出し結果を表示します。
    成功した場合、生成されたタイトルはセッションステートに保存されます。

    Args:
        prefectures (list): 都道府県名のリスト。
    """

    if st.button(
        f"{selected_prefecture_name}のタイトルを生成する", key="generate_titles_button"
    ):
        # タイトル生成ボタンが押されたら、関連するセッションステートをリセット
        st.session_state.titles_generated_successfully = False
        st.session_state.main_title_generated = None
        st.session_state.sub_titles_generated = None

        with st.spinner(
            f"{selected_prefecture_name}の情報を検索し、タイトルを生成しています..."
        ):
            result_titles = generate_titles_for_prefecture(selected_prefecture_name)

        st.subheader("タイトル生成結果")
        if result_titles.get("error"):  # エラーキーが存在し、かつ真の場合
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
            st.success("タイトルの生成に成功しました！")
            generated_main_title = result_titles["titles_output"]["main_title"]
            generated_sub_titles = result_titles["titles_output"]["sub_titles"]

            st.markdown(f"**メインタイトル:** {generated_main_title}")
            st.markdown("**サブタイトル:**")
            for i, sub_title in enumerate(generated_sub_titles):
                st.markdown(f"- {sub_title}")

            # 生成されたタイトルをセッションステートに保存
            st.session_state.main_title_generated = generated_main_title
            st.session_state.sub_titles_generated = generated_sub_titles
            st.session_state.titles_generated_successfully = True
        else:
            st.warning(
                "タイトルを生成できませんでした。APIからの応答に必要な情報が含まれていません。"
            )
            st.session_state.titles_generated_successfully = False


def render_article_generation_section():
    """
    記事生成のためのUIコンポーネントを描画し、関連するロジックを実行します。
    タイトルが正常に生成されている場合にのみ、記事生成ボタンが表示されます。
    ボタンが押されると、セッションステートに保存されたタイトルを使用して記事を生成し、結果を表示します。
    """
    if st.session_state.get("titles_generated_successfully") and st.session_state.get(
        "main_title_generated"
    ):
        st.markdown("---")  # タイトル生成結果と記事生成ボタンの間に区切り線
        if st.button(
            "📝 上記のタイトルで記事を生成する",
            type="primary",
            use_container_width=True,
            key="generate_article_button",
        ):
            main_title_for_article = st.session_state.main_title_generated
            sub_titles_for_article = st.session_state.sub_titles_generated

            if not main_title_for_article or not sub_titles_for_article:
                st.error(
                    "記事を生成するためのタイトル情報が見つかりません。再度タイトルを生成してください。"
                )
                return  # 処理を中断

            with st.spinner(f"「{main_title_for_article}」の記事を生成中です..."):
                article_result = generate_article_workflow(
                    main_title_for_article, sub_titles_for_article
                )

            st.markdown("---")
            st.subheader("📄 記事生成結果")

            displayed_topic = article_result.get("main_title", main_title_for_article)

            if article_result.get("success"):
                st.success(f"🎉 記事「{displayed_topic}」の生成に成功しました！")
                article_content_preview = article_result.get("html_output")
                if article_content_preview:
                    st.html(article_content_preview)
                else:
                    st.warning("生成された記事のHTMLコンテンツがありません。")
            else:
                st.error(f"❌ 記事「{displayed_topic}」の生成に失敗しました。")
                if article_result.get("error_message"):
                    st.error(
                        f"**エラーメッセージ:** {article_result.get('error_message')}"
                    )
                error_html_output = article_result.get("html_output")
                if error_html_output:  # エラー時でもHTML出力があれば表示
                    st.markdown("#### エラー時のHTML出力プレビュー:")
                    st.html(error_html_output)


def article_generator_app(selected_prefecture_name):
    """
    AI記事ジェネレーターアプリケーションのメイン関数。
    ページ設定、セッションステート初期化、各UIセクションの描画を行います。
    """

    st.title("✍️ AI記事ジェネレーター")
    st.markdown("---")

    initialize_session_state()
    render_title_generation_section(selected_prefecture_name)
    render_article_generation_section()

    # アプリケーション下部に注意書きなど（オプション）
    st.markdown("---")
    st.caption("AI記事ジェネレーター v1.0")



