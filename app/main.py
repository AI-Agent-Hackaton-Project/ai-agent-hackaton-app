import streamlit as st
import vertexai
from vertexai.language_models import ChatModel, InputOutputTextPair

def main():
    """
    Streamlitアプリケーションのメイン処理を行う関数
    """
    st.title('シンプルなStreamlitアプリ')

    import os

    PROJECT_ID = "linen-option-411401"
    LOCATION = "us-central1"
    MODEL_NAME = "gemini-1.0-pro" 

    try:
        # 環境変数からプロジェクトIDを取得しようと試みる（デプロイ環境向け）
        project_id_env = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if project_id_env:
            PROJECT_ID = project_id_env

        if not PROJECT_ID or PROJECT_ID == "your-gcp-project-id":
            st.error("GCPプロジェクトIDが設定されていません。コード内の PROJECT_ID を設定するか、環境変数 GOOGLE_CLOUD_PROJECT を設定してください。")
            st.stop()

        vertexai.init(project=PROJECT_ID, location=LOCATION)
        st.info(f"Vertex AI SDK initialized. Project: {PROJECT_ID}, Location: {LOCATION}")

    except Exception as e:
        st.error(f"Vertex AI の初期化に失敗しました: {e}")
        st.info("認証情報を確認してください。ローカル環境では 'gcloud auth application-default login' を実行しましたか？")
        st.stop()

    # --- Streamlit アプリケーション ---
    st.title("💬 Vertex AI チャットアプリ")
    st.caption(f"Powered by Google Vertex AI ({MODEL_NAME})")

    # --- チャットセッションの初期化 ---
    # st.session_stateを使ってチャット履歴とVertex AIのチャットセッションを管理
    if "messages" not in st.session_state:
        st.session_state.messages = [] # チャット履歴を格納するリスト
    if "chat_session" not in st.session_state:
        try:
            # チャットモデルをロード
            chat_model = ChatModel.from_pretrained(MODEL_NAME)
            # チャットセッションを開始
            # context: 必要であれば、AIに前提知識を与えるためのコンテキストを設定
            # examples: Few-shotプロンプティングのための例を設定
            st.session_state.chat_session = chat_model.start_chat(
                context="あなたは親切で役立つAIアシスタントです。",
                # examples=[
                #     InputOutputTextPair(
                #         input_text="ユーザーからの入力例",
                #         output_text="期待するAIの応答例",
                #     ),
                # ]
            )
        except Exception as e:
            st.error(f"チャットモデル ({MODEL_NAME}) のロードまたはセッションの開始に失敗しました: {e}")
            st.stop()

    # --- チャット履歴の表示 ---
    # st.session_state.messages に保存されている履歴を表示
    for message in st.session_state.messages:
        # "role" (user or model) に基づいて表示を分ける
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- ユーザー入力の処理 ---
    # st.chat_inputでユーザーからの入力を受け付ける
    if prompt := st.chat_input("メッセージを入力してください..."):
        # ユーザーのメッセージを履歴に追加し、表示
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- Vertex AI への問い合わせと応答処理 ---
        try:
            # Vertex AIのチャットセッションにメッセージを送信
            chat = st.session_state.chat_session
            # パラメータで応答を制御することも可能
            # parameters = {
            #     "temperature": 0.2,       # 応答のランダム性 (0に近いほど決定的)
            #     "max_output_tokens": 256, # 最大出力トークン数
            #     "top_p": 0.8,             # Top-pサンプリング
            #     "top_k": 40,              # Top-kサンプリング
            # }
            # response = chat.send_message(prompt, **parameters)
            response = chat.send_message(prompt)
            response_text = response.text

            # AIの応答を履歴に追加し、表示
            st.session_state.messages.append({"role": "model", "content": response_text})
            with st.chat_message("assistant"): # StreamlitではAIの役割は"assistant"が一般的
                st.markdown(response_text)

        except Exception as e:
            st.error(f"Vertex AI への問い合わせ中にエラーが発生しました: {e}")
            st.warning("もう一度試すか、少し待ってから再度入力してください。")


if __name__ == "__main__":
    main()