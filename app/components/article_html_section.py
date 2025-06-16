import streamlit as st
import traceback

from utils.agent_generate_article import generate_article_workflow
from utils.generate_titles import generate_titles_for_prefecture


def initialize_session_state():
    """
    セッション状態を初期化します。
    """
    if "main_title_generated" not in st.session_state:
        st.session_state.main_title_generated = None
    if "sub_titles_generated" not in st.session_state:
        st.session_state.sub_titles_generated = None
    if "titles_generated_successfully" not in st.session_state:
        st.session_state.titles_generated_successfully = False


def improve_html_styling(html_content):
    """
    HTMLコンテンツのスタイリングを改善し、ハイライトを減らして読みやすくします。
    """
    if not html_content:
        return html_content

    # より記事らしいスタイルのCSS
    improved_css = """
    <style>
    .article-container {
        max-width: 800px;
        margin: 0 auto;
        font-family: 'Hiragino Sans', 'Noto Sans JP', 'Yu Gothic', sans-serif;
        line-height: 1.8;
        color: #333;
        background: #fafafa;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .article-container h1 {
        color: #2c3e50;
        font-size: 2.2em;
        margin-bottom: 20px;
        text-align: center;
        border-bottom: 3px solid #6f92a9;
        padding-bottom: 15px;
        font-weight: 700;
    }
    
    .article-container h2 {
        color: #34495e;
        font-size: 1.6em;
        margin: 35px 0 20px 0;
        padding-left: 15px;
        background: linear-gradient(90deg, #f8f9fa 0%, transparent 100%);
        padding: 15px;
        border-radius: 5px;
    }
    
    .article-container h3 {
        color: #2c3e50;
        font-size: 1.3em;
        margin: 25px 0 15px 0;
        padding-bottom: 8px;
        border-bottom: 2px dotted #bdc3c7;
    }
    
    .article-container p {
        margin-bottom: 18px;
        text-align: justify;
    }
    
    .article-container ul, .article-container ol {
        margin: 20px 0;
        padding-left: 30px;
    }
    
    .article-container li {
        margin-bottom: 8px;
        line-height: 1.7;
    }
    
    /* ハイライトを控えめに - 重要な箇所のみ */
    .article-container mark, .article-container .highlight {
        background: linear-gradient(transparent 60%, #fff3cd 60%);
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: 500;
        color: #856404;
    }
    
    /* 強調テキスト */
    .article-container strong {
        color: #2c3e50;
        font-weight: 700;
    }
    
    /* 引用風のスタイル */
    .article-container blockquote {
        border-left: 4px solid #3498db;
        padding: 15px 20px;
        margin: 20px 0;
        background: #ecf0f1;
        font-style: italic;
        border-radius: 0 8px 8px 0;
    }
    
    /* 画像のスタイリング */
    .article-container img {
        max-width: 100%;
        height: auto;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.15);
        margin: 20px 0;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* 段落間の空白を調整 */
    .article-container p + p {
        margin-top: 1.2em;
    }
    
    /* レスポンシブ対応 */
    @media (max-width: 768px) {
        .article-container {
            padding: 20px;
            margin: 10px;
        }
        
        .article-container h1 {
            font-size: 1.8em;
        }
        
        .article-container h2 {
            font-size: 1.4em;
        }
    }
    </style>
    """

    # HTMLからbodyタグの内容を抽出
    if "<body>" in html_content and "</body>" in html_content:
        body_start = html_content.find("<body>") + 6
        body_end = html_content.find("</body>")
        body_content = html_content[body_start:body_end]
    else:
        body_content = html_content

    # 過度なハイライトを削除（3つ以上連続するハイライトを減らす）
    import re

    # <mark>タグの数を制限
    mark_pattern = r"<mark[^>]*>(.*?)</mark>"
    marks = re.findall(mark_pattern, body_content)

    # 短い単語（3文字以下）のハイライトを削除
    def reduce_highlights(match):
        content = match.group(1)
        if len(content) <= 3 or content in [
            "は",
            "が",
            "を",
            "に",
            "で",
            "と",
            "から",
            "まで",
        ]:
            return content
        return match.group(0)

    body_content = re.sub(mark_pattern, reduce_highlights, body_content)

    # 改行とスペーシングを改善
    body_content = re.sub(r"\n\s*\n", "\n\n", body_content)  # 複数の空行を2つに
    body_content = re.sub(r"。\s*", "。<br>", body_content)  # 句点後に改行

    # 完全なHTMLを構築
    improved_html = f"""
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>記事</title>
        {improved_css}
    </head>
    <body>
        <div class="article-container">
            {body_content}
        </div>
    </body>
    </html>
    """

    return improved_html


def render_title_generation_section(selected_prefecture_name):
    """
    タイトル生成と記事生成の全プロセスを管理し、画像生成も含めたプログレスバーで進捗を表示します。
    """
    if st.button(
        f"{selected_prefecture_name}のタイトルと記事を生成する",
        key="generate_titles_and_article_button",
    ):
        # --- UIと変数の初期化 ---
        st.session_state.titles_generated_successfully = False
        st.session_state.main_title_generated = None
        st.session_state.sub_titles_generated = None

        final_state = None
        error_occurred = False
        error_message = ""

        estimated_sub_images = (
            len(generated_sub_titles)
            if st.session_state.get("sub_titles_generated")
            else 5
        )
        total_steps = 8 + estimated_sub_images
        completed_steps = 0

        st.markdown("---")
        # カラムを使ってタイトルと進行状況テキストを横に並べる
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write("### 全体進行率")

        # メインのstatusコンテナ。この中で全ての処理が行われる。
        with st.status("生成を準備中です…", expanded=True) as status:

            progress_bar = st.progress(0)
            progress_text_placeholder = st.empty()

            # 現在の処理内容を表示するプレースホルダー
            current_process_placeholder = st.empty()

            # 初期値を設定
            progress_text_placeholder.text(f"0 / {total_steps} (0%)")

            try:
                # --- ステップ1: タイトル生成 ---
                completed_steps += 1
                message = f"ステップ {completed_steps}/{total_steps}: {selected_prefecture_name}のタイトルを生成しています…"
                status.update(label=message)
                progress_value = completed_steps / total_steps
                percentage = progress_value * 100
                progress_text_placeholder.text(f"進行率: {percentage:.0f}%")
                current_process_placeholder.info("📝 タイトルを考案中...")
                progress_bar.progress(progress_value)

                result_titles = generate_titles_for_prefecture(selected_prefecture_name)

                if result_titles.get("error"):
                    # タイトル生成でエラーが発生した場合
                    error_occurred = True
                    error_message = result_titles["error"]
                    st.error(f"タイトル生成中にエラー: {error_message}")
                    st.json(result_titles.get("details", "詳細不明"))
                    if "raw_response" in result_titles:
                        st.text_area(
                            "LLM Raw Response:",
                            result_titles["raw_response"],
                            height=200,
                        )
                    status.update(label="タイトル生成エラー", state="error")
                    progress_text_placeholder.text("エラー発生")
                    current_process_placeholder.error(
                        "❌ タイトル生成でエラーが発生しました"
                    )

                elif result_titles.get("titles_output"):
                    # タイトル生成成功
                    st.session_state.titles_generated_successfully = True
                    generated_main_title = result_titles["titles_output"]["main_title"]
                    generated_sub_titles = result_titles["titles_output"]["sub_titles"]
                    st.session_state.main_title_generated = generated_main_title
                    st.session_state.sub_titles_generated = generated_sub_titles

                    current_process_placeholder.success(
                        f"✅ タイトル生成完了: {generated_main_title}"
                    )

                    # --- ステップ2以降: 記事生成 ---
                    stream = generate_article_workflow(
                        generated_main_title,
                        generated_sub_titles,
                        selected_prefecture_name,
                    )

                    for event in stream:
                        # 画像生成のステップを特別に処理
                        step_name = event.get("step")
                        step_message = event.get("message", "処理中…")
                        image_progress = event.get(
                            "image_progress"
                        )  # 画像生成の詳細進捗

                        # 画像生成ステップの詳細処理
                        if image_progress:
                            # 詳細な画像生成進捗を表示
                            if image_progress.get("type") == "main_image":
                                completed_steps += 1
                                current_process_placeholder.info(
                                    f"🎨 4コマ画像を生成中..."
                                )
                            elif image_progress.get("type") == "subtitle_image":
                                current_index = image_progress.get("current", 0)
                                total_count = image_progress.get("total", 0)
                                subtitle = image_progress.get("subtitle", "")
                                completed_steps += 1
                                current_process_placeholder.info(
                                    f"🖼️ サブ画像生成中 [{current_index}/{total_count}]: 「{subtitle}」"
                                )
                            elif image_progress.get("type") == "subtitle_image_start":
                                total_count = image_progress.get("total", 0)
                                current_process_placeholder.info(
                                    f"🖼️ サブタイトル用画像生成開始 (全{total_count}個)"
                                )
                            elif (
                                image_progress.get("type") == "subtitle_image_complete"
                            ):
                                total_count = image_progress.get("total", 0)
                                current_process_placeholder.success(
                                    f"✅ サブタイトル用画像生成完了 ({total_count}個)"
                                )
                        elif step_name and ("subtitle_images_item" in step_name):
                            # 個別のサブタイトル画像生成ステップ
                            completed_steps += 1
                            current_process_placeholder.info(f"🔄 {step_message}")
                        elif "画像生成" in step_message or (
                            step_name and "image" in step_name.lower()
                        ):
                            # 従来の画像生成メッセージの処理
                            completed_steps += 1
                            current_process_placeholder.info(f"🎨 {step_message}")
                        else:
                            completed_steps += 1
                            current_process_placeholder.info(f"🔄 {step_message}")

                        if "error" in event:
                            error_occurred = True
                            error_message = event["error"]
                            final_state = event.get("state", final_state)
                            status.update(
                                label="記事生成エラー", state="error", expanded=False
                            )
                            progress_text_placeholder.text("エラー発生")
                            current_process_placeholder.error(
                                f"❌ エラー: {error_message}"
                            )
                            progress_bar.progress(1.0)
                            break

                        message = (
                            f"ステップ {completed_steps}/{total_steps}: {step_message}"
                        )
                        final_state = event.get("state")

                        status.update(label=message)
                        progress_value = min(1.0, completed_steps / total_steps)
                        percentage = progress_value * 100
                        progress_text_placeholder.text(f"進行率: {percentage:.0f}%")

                        progress_bar.progress(progress_value)

                        if step_name == "__end__":
                            status.update(
                                label="完了しました！", state="complete", expanded=False
                            )
                            progress_text_placeholder.text("進行率: 100%")
                            current_process_placeholder.success(
                                "🎉 すべての処理が完了しました！"
                            )
                            progress_bar.progress(1.0)
                            break
                else:
                    # titles_outputがないという稀なケース
                    error_occurred = True
                    error_message = "タイトルを生成できませんでした。APIからの応答が予期した形式ではありません。"
                    st.warning(error_message)
                    status.update(label="タイトル取得エラー", state="error")
                    current_process_placeholder.error("❌ タイトルの取得に失敗しました")

            except Exception as e:
                error_occurred = True
                error_message = f"予期せぬエラーが発生しました: {e}"
                status.update(label="予期せぬエラー", state="error")
                progress_text_placeholder.text("予期せぬエラー")
                current_process_placeholder.error(f"❌ 予期せぬエラー: {e}")
                traceback._exc()

        # --- 最終結果の表示 ---
        if error_occurred:
            st.error(
                f"❌ 「{selected_prefecture_name}」に関する記事の生成に失敗しました。"
            )
            if error_message:
                st.error(f"**エラーメッセージ:** {error_message}")
            if final_state and final_state.get("html_output"):
                st.markdown("#### エラー発生時のHTMLプレビュー:")
                st.html(final_state.get("html_output"))
        elif final_state:
            html_output = final_state.get("html_output")
            if html_output:
                # HTMLスタイリングを改善
                improved_html = improve_html_styling(html_output)
                st.html(improved_html)
            else:
                st.warning("生成された記事のHTMLコンテンツがありませんでした。")
        else:
            if not error_occurred:  # エラーではないのに結果がない場合
                st.error(
                    "記事生成プロセスは完了しましたが、最終的な結果を取得できませんでした。"
                )


def article_generator_app(selected_prefecture_name):
    """
    記事生成アプリのメイン関数
    """
    initialize_session_state()
    render_title_generation_section(selected_prefecture_name)
