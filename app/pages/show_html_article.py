import streamlit as st
import os
from utils.agent_generate_article import generate_article_workflow
from utils.generate_titles import generate_titles_for_prefecture
from config.constants import JAPAN_PREFECTURES

# --- Streamlit アプリケーション ---
st.set_page_config(page_title="記事ジェネレーター 📝", layout="wide")

st.title("✍️ AI記事ジェネレーター")
st.markdown("---")

if "main_title_generated" not in st.session_state:
    st.session_state.main_title_generated = None
if "sub_titles_generated" not in st.session_state:
    st.session_state.sub_titles_generated = None
if "titles_generated_successfully" not in st.session_state:
    st.session_state.titles_generated_successfully = False


selected_prefecture_name = st.selectbox(
    "タイトルを生成したい都道府県を選択してください:",
    JAPAN_PREFECTURES,
    key="prefecture_selectbox",
)

if st.button(
    f"{selected_prefecture_name}のタイトルを生成する", key="generate_titles_button"
):
    st.session_state.titles_generated_successfully = False
    st.session_state.main_title_generated = None
    st.session_state.sub_titles_generated = None

    with st.spinner(
        f"{selected_prefecture_name}の情報を検索し、タイトルを生成しています..."
    ):
        result_titles = generate_titles_for_prefecture(selected_prefecture_name)

    st.subheader("タイトル生成結果")
    if "error" in result_titles and result_titles["error"]:
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
        st.warning("タイトルを生成できませんでした。")
        st.session_state.titles_generated_successfully = False

# タイトルが正常に生成された場合のみ記事生成ボタンを表示
if (
    st.session_state.titles_generated_successfully
    and st.session_state.main_title_generated
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
        else:
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
                # エラー時にもHTML出力がある場合（エラーメッセージを含むHTMLなど）は表示
                error_html_output = article_result.get("html_output")
                if error_html_output:
                    st.markdown("#### エラー時のHTML出力プレビュー:")
                    st.html(error_html_output)


# アプリケーション下部に注意書きなど（オプション）
st.markdown("---")
st.caption("AI記事ジェネレーター v1.0")
