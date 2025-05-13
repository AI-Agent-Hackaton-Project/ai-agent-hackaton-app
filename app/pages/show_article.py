from vertexai.generative_models import GenerativeModel
import streamlit as st
from config.config import get_config


def show_article():

    config = get_config()

    st.title("💬 Vertex AI チャットアプリ")

    # --- チャットセッションの初期化 ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_session" not in st.session_state:
        model = GenerativeModel(config["model_name"])
        st.session_state.chat_session = model.start_chat()

    # --- チャット履歴の表示 ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- ユーザー入力の処理 ---
    if prompt := st.chat_input("メッセージを入力してください..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            chat = st.session_state.chat_session
            response = chat.send_message(prompt)
            response_text = response.text

            st.session_state.messages.append(
                {"role": "assistant", "content": response_text}
            )
            with st.chat_message("assistant"):
                st.markdown(response_text)

        except Exception as e:
            st.error(f"Vertex AI への問い合わせ中にエラーが発生しました: {e}")
            st.warning("もう一度試すか、少し待ってから再度入力してください。")

show_article()