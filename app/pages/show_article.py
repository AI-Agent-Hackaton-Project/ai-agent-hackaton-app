from utils.generate_article import generate_article
import streamlit as st
from config.constants import JAPAN_PREFECTURES
import traceback


st.set_page_config(page_title="観光記事ジェネレーター", layout="wide")
st.title("AI観光記事ジェネレーター 📝")
st.markdown("都道府県を選択すると、AIがその地域の歴史に関する記事を自動生成します。")

with st.sidebar:
    st.header("設定")

    selected_prefecture = st.selectbox(
        "都道府県を選択:",
        JAPAN_PREFECTURES,
        key="sidebar_prefecture",
        index=0,
    )

if st.button(f"{selected_prefecture} の記事を生成する ✨", type="primary"):
    if not selected_prefecture:
        st.warning("都道府県を選択してください。")
    else:
        st.info(f"「{selected_prefecture}」の記事生成を開始します...")  # 処理開始を通知
        article_data = None  # article_dataを初期化
        with st.spinner(
            f"AIが {selected_prefecture} の歴史に関する記事を執筆中です... しばらくお待ちください ⏳"
        ):
            try:
                article_data = generate_article(selected_prefecture)
                st.success("generate_article関数が完了しました。")  # 完了したことを通知

                # --- ここからデバッグ情報表示 ---
                st.subheader("デバッグ情報: generate_articleからの返り値")
                if isinstance(article_data, dict):
                    st.json(article_data)
                elif article_data is None:
                    st.write("`generate_article` から `None` が返されました。")
                else:
                    st.write(f"返り値の型: {type(article_data)}")
                    st.write(f"返り値の内容: {str(article_data)}")
                st.write("--- デバッグ情報ここまで ---")
                # --- ここまでデバッグ情報表示 ---

            except Exception as e:
                st.error(
                    f"generate_article関数呼び出し中に予期せぬエラーが発生しました: {e}"
                )
                st.text_area(
                    "エラー詳細 (Traceback)", traceback.format_exc(), height=300
                )
                # エラーが発生した場合、article_dataにエラー情報を含めて後続処理でエラー表示できるようにする
                article_data = {
                    "error": "Exception in generate_article",
                    "details": str(e),
                    "raw_traceback": traceback.format_exc(),
                }

        if article_data is not None:  # Noneでないことを確認
            if "error" not in article_data:
                title_text = article_data.get("title")
                if not title_text:
                    title_text = f"{selected_prefecture}の歴史について"
                st.subheader(f"🎉 生成された記事: {title_text}")
                st.divider()

                if "block" in article_data and isinstance(article_data["block"], list):
                    st.markdown(f"### 記事本文")
                    for i, block_content in enumerate(article_data["block"]):
                        st.markdown(f"#### 第 {i+1} ブロック")
                        st.markdown(
                            block_content
                            if block_content
                            and block_content.strip()  # Noneや空文字列でないことを確認
                            else "このブロックには内容が記述されていません。"
                        )
                        if i < len(article_data["block"]) - 1:
                            st.markdown("---")
                else:
                    st.warning(
                        "記事のブロックデータが見つからないか、形式が正しくありません。LLMの出力形式を確認してください。"
                    )

                st.divider()
                st.success(f"{selected_prefecture} の記事生成が完了しました！✅")

                with st.expander("生成されたJSONデータを見る (RAW) 🔍"):
                    st.json(article_data)
            else:
                # エラー情報がarticle_dataに含まれている場合の表示
                st.error(
                    f"記事生成中にエラーが発生しました: {article_data.get('error', '不明なエラー')}"
                )
                if "details" in article_data:
                    st.warning(f"詳細: {article_data['details']}")
                if (
                    "raw_traceback" in article_data
                ):  # try-exceptで補足したトレースバック
                    with st.expander("エラーのトレースバックを見る"):
                        st.text_area("", article_data["raw_traceback"], height=200)
                elif (
                    "raw_response" in article_data
                ):  # generate_article内部でセットされたraw_response
                    with st.expander("LLMからの生のレスポンスを見る"):
                        st.text_area("", article_data["raw_response"], height=200)

        else:
            st.error(
                "記事データの生成に失敗しました。`generate_article` 関数から有効な応答がありませんでした。"
            )

st.markdown("---")
st.caption("このAI記事ジェネレーターは Vertex AI を利用しています。")
