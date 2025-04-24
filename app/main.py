import streamlit as st
import vertexai
from vertexai.language_models import ChatModel
from dotenv import load_dotenv
import os

def main():
    st.title('シンプルなStreamlitアプリ')

    load_dotenv()

    gcp_project_id = os.getenv("GCP_PROJECT_ID")
    gcp_location = os.getenv("GCP_LOCATION")
    model_name = os.getenv("VERTEX_AI_MODEL_NAME")

    vertexai.init(project=gcp_project_id, location=gcp_location)

    # --- Streamlit アプリケーション ---
    st.title("💬 Vertex AI チャットアプリ")
    st.caption(f"Powered by Google Vertex AI ({gcp_location})")

    # --- チャットセッションの初期化 ---
    if "messages" not in st.session_state:
        st.session_state.messages = [] 
    if "chat_session" not in st.session_state:
        try:
            chat_model = ChatModel.from_pretrained(model_name)
            st.session_state.chat_session = chat_model.start_chat(
                context="あなたは親切で役立つAIアシスタントです。",
            )
        except Exception as e:
            st.error(f"チャットモデル ({model_name}) のロードまたはセッションの開始に失敗しました: {e}")
            st.stop()

    # --- チャット履歴の表示 ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- ユーザー入力の処理 ---
    if prompt := st.chat_input("メッセージを入力してください..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- Vertex AI への問い合わせと応答処理 ---
        try:
            chat = st.session_state.chat_session
            response = chat.send_message(prompt)
            response_text = response.text

            st.session_state.messages.append({"role": "model", "content": response_text})
            with st.chat_message("assistant"): 
                st.markdown(response_text)

        except Exception as e:
            st.error(f"Vertex AI への問い合わせ中にエラーが発生しました: {e}")
            st.warning("もう一度試すか、少し待ってから再度入力してください。")


if __name__ == "__main__":
    main()
