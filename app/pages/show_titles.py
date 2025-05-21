from config.constants import JAPAN_PREFECTURES
import streamlit as st
from utils.generate_titles import generate_titles_for_prefecture


# --- Streamlit UI (呼び出し例) ---
if __name__ == "__main__":
    st.title("都道府県別 SEOタイトルジェネレーター")

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

        st.subheader("検索結果（参考情報）")
        if result.get("search_results_for_display"):
            for i, sr in enumerate(result["search_results_for_display"]):
                st.markdown(f"**{i+1}. {sr.get('title', 'タイトルなし')}**")
                st.markdown(
                    f"   [{sr.get('link', 'リンクなし')}]({sr.get('link', 'リンクなし')})"
                )
                st.caption(f"   {sr.get('snippet', 'スニペットなし')}")
        else:
            st.info("表示する検索結果はありませんでした。")
