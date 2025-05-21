import streamlit as st
import os
from utils.agent_generate_article import generate_article_workflow
from config.constants import JAPAN_PREFECTURES


# --- Streamlit アプリケーション ---
st.set_page_config(page_title="記事ジェネレーター 📝", layout="wide")

st.title("✍️ AI記事ジェネレーター")
st.markdown("---")

with st.sidebar:
    st.header("設定")

    user_topic = st.selectbox(
        "都道府県を選択:",
        JAPAN_PREFECTURES,
        key="sidebar_prefecture",
        index=0,
    )


current_script_dir = os.path.dirname(os.path.abspath(__file__))

# 記事生成ボタン
if st.button("📝 記事を生成する", type="primary", use_container_width=True):
    if not user_topic.strip():
        st.warning("⚠️ トピックが入力されていません。")
    else:
        with st.spinner(f"「{user_topic}」に関する記事を生成中です..."):
            result = generate_article_workflow(user_topic)

        st.markdown("---")
        st.subheader("📄 生成結果")
        if result.get("success"):
            st.success(f"🎉 記事の生成に成功しました！")
            st.info(f"**トピック:** {result.get('topic')}")
            st.info(f"**出力ファイルパス:** `{result.get('output_file_path')}`")

            # 生成された記事の内容をプレビュー表示 (オプション)
            try:
                article_content_preview = result.get("html_output")
                st.html(article_content_preview)
            except Exception as e:
                st.warning(f"⚠️ 記事ファイルの読み込み中にエラーが発生しました: {e}")

        else:
            st.error(f"❌ 記事の生成に失敗しました。")
            st.info(f"**トピック:** {result.get('topic')}")
            if result.get("error_message"):
                st.error(f"**エラーメッセージ:** {result.get('error_message')}")

st.markdown("---")
st.caption("© 2024 AI Article Generator")
