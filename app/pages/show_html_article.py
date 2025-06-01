import streamlit as st
import os
from utils.agent_generate_article import generate_article_workflow
from utils.generate_titles import generate_titles_for_prefecture
from config.constants import JAPAN_PREFECTURES


# --- Streamlit アプリケーション ---
st.set_page_config(page_title="記事ジェネレーター 📝", layout="wide")

st.title("✍️ AI記事ジェネレーター")
st.markdown("---")

selected_prefecture_name = st.selectbox(
    "タイトルを生成したい都道府県を選択してください:", JAPAN_PREFECTURES
)

if st.button(f"{selected_prefecture_name}のタイトルを生成する"):
    with st.spinner(
        f"{selected_prefecture_name}の情報を検索し、タイトルを生成しています..."
    ):
        result = generate_titles_for_prefecture(selected_prefecture_name)

    st.subheader("生成結果")
    if "error" in result and result["error"]:
        st.error(f"エラーが発生しました: {result['error']}")
        st.json(result.get("details", "詳細不明"))
        if "raw_response" in result:
            st.text_area(
                "LLM Raw Response (on error):", result["raw_response"], height=200
            )
    elif result.get("titles_output"):
        st.success("タイトルの生成に成功しました！")
        st.markdown(f"**メインタイトル:** {result['titles_output']['main_title']}")
        st.markdown("**サブタイトル:**")
        for i, sub_title in enumerate(result["titles_output"]["sub_titles"]):
            st.markdown(f"- {sub_title}")
    else:
        st.warning("タイトルを生成できませんでした。")
