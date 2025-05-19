from utils.generate_article import generate_article
import streamlit as st
from config.constants import JAPAN_PREFECTURES
import traceback


st.set_page_config(page_title="観光記事ジェネレーター", layout="wide")
st.title("AI観光記事ジェネレーター 📝")
st.markdown("都道府県を選択すると、AIがその地域の歴史に関する記事を自動生成します。")

with st.sidebar:
    st.header("設定")

    if not JAPAN_PREFECTURES:  # JAPAN_PREFECTURESが空でないかチェック
        st.error("都道府県リストが constants.py から正しく読み込めませんでした。")
        st.stop()

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
        st.info(f"「{selected_prefecture}」の記事生成を開始します...")
        article_generation_result = None  # result を先に定義
        with st.spinner(
            f"AIが {selected_prefecture} の歴史に関する記事を執筆中です... しばらくお待ちください ⏳"
        ):
            try:
                article_generation_result = generate_article(selected_prefecture)
                st.success("generate_article関数が完了しました。")

                # --- デバッグ情報表示 ---
                st.subheader("デバッグ情報: generate_articleからの返り値")
                if isinstance(article_generation_result, dict):
                    st.json(article_generation_result)
                elif article_generation_result is None:
                    st.write("`generate_article` から `None` が返されました。")
                else:
                    st.write(f"返り値の型: {type(article_generation_result)}")
                    st.write(f"返り値の内容: {str(article_generation_result)}")
                st.write("--- デバッグ情報ここまで ---")

            except Exception as e:
                st.error(
                    f"generate_article関数呼び出し中に予期せぬエラーが発生しました: {e}"
                )
                st.text_area(
                    "エラー詳細 (Traceback)", traceback.format_exc(), height=300
                )
                article_generation_result = {  # エラー情報を格納
                    "error": "Exception in UI calling generate_article",
                    "details": str(e),
                    "raw_traceback": traceback.format_exc(),
                }

        if article_generation_result is not None:
            # 検索結果はエラーの有無に関わらず取得を試みる
            search_results_display = article_generation_result.get(
                "search_results_for_display"
            )

            if "error" not in article_generation_result:
                # 記事コンテンツの取得
                article_content = article_generation_result.get("article_content")

                if article_content:  # article_content が None でないことを確認
                    title_text = article_content.get("title")
                    if not title_text:
                        title_text = (
                            f"{selected_prefecture}の歴史について"  # フォールバック
                        )
                    st.subheader(f"🎉 生成された記事: {title_text}")
                    st.divider()

                    if "block" in article_content and isinstance(
                        article_content.get("block"), list
                    ):
                        st.markdown(f"### 記事本文")
                        for i, block_item_content in enumerate(
                            article_content["block"]
                        ):
                            st.markdown(f"#### 第 {i+1} ブロック")
                            st.markdown(
                                block_item_content
                                if block_item_content and block_item_content.strip()
                                else "このブロックには内容が記述されていません。"
                            )
                            if i < len(article_content["block"]) - 1:
                                st.markdown("---")
                    else:
                        st.warning(
                            "記事のブロックデータが見つからないか、形式が正しくありません。"
                        )
                    st.divider()
                    st.success(f"{selected_prefecture} の記事生成が完了しました！✅")

                    # 生成された記事JSONデータを表示
                    with st.expander("生成された記事JSONデータを見る (RAW) 🔍"):
                        st.json(article_content)  # 記事コンテンツのJSON
                else:
                    st.warning(
                        "記事コンテンツが取得できませんでした。"
                    )  # article_content が None の場合

                # 検索結果の表示 (成功時)
                if search_results_display:
                    with st.expander("参照されたGoogle検索結果を見る", expanded=False):
                        st.markdown(search_results_display)
                else:
                    st.info("表示する検索結果はありませんでした。")

            else:  # エラーがある場合
                st.error(
                    f"記事生成中にエラーが発生しました: {article_generation_result.get('error', '不明なエラー')}"
                )
                if "details" in article_generation_result:
                    st.warning(f"詳細: {article_generation_result['details']}")
                if "raw_traceback" in article_generation_result:
                    with st.expander("エラーのトレースバックを見る"):
                        st.text_area(
                            "", article_generation_result["raw_traceback"], height=200
                        )
                elif (
                    "raw_response" in article_generation_result
                ):  # OutputParserExceptionの場合など
                    with st.expander("LLMからの生のレスポンスを見る"):
                        st.text_area(
                            "", article_generation_result["raw_response"], height=200
                        )

                # 検索結果の表示 (エラー時も、検索試行結果があれば表示)
                if search_results_display:
                    with st.expander(
                        "Google検索結果 (エラー発生時の情報)", expanded=True
                    ):
                        st.markdown(search_results_display)
                else:
                    st.info("エラー発生時の検索関連情報はありませんでした。")
        else:
            st.error(
                "記事データの生成に失敗しました。`generate_article` 関数から有効な応答がありませんでした。"
            )

st.markdown("---")
st.caption(
    "このAI記事ジェネレーターは Vertex AI と Vertex AI Search and Conversation を利用しています。"
)  # 少し変更
