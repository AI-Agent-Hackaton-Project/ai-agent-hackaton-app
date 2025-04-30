import streamlit as st
from components.simple_chatbot import show_simple_chatbot
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, load_prompt
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages.chat import ChatMessage


def main():
    load_dotenv()

    # 会話履歴にメッセージを追加する関数（ChatMessage形式で追加）
    def add_history(role, content):
        st.session_state.messages.append(ChatMessage(role=role, content=content))

    # 会話履歴を画面に出力する関数
    def print_history():
        for msg in st.session_state["messages"]:
            st.chat_message(msg.role).write(msg.content)

    # プロンプトの種類によって、適切なチェーンを作成
    def create_chain(prompt_type="基本"):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "回答を簡潔にしてください。",
                ),
                ("user", "#Question:\n{question}"),
            ]
        )
        if prompt_type == "SNS":
            prompt = load_prompt("app/prompts/sns.yaml")
        elif prompt_type == "まとめ":
            prompt = load_prompt("app/prompts/summarize.yaml")

        model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, streaming=True)

        chain = prompt | model | StrOutputParser()
        return chain

    st.title("シンプルなStreamlit Chatアプリ")
    st.info(
        "このアプリは、StreamlitとLangChainを使用して構築されたシンプルなチャットボットです。"
    )
    # セッションステートに履歴がなければ初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # サイドバーのUI（会話履歴のクリアボタンとプロンプト種類の選択）
    with st.sidebar:
        clear_button = st.button("会話をクリア")
        if clear_button:
            st.session_state.messages = []

        selected_option = st.selectbox(
            "カテゴリーを選択してください",
            ("基本", "SNS", "まとめ"),
            index=0,
            help="選択したカテゴリーに基づいて、AIの応答が変わります。",
        )

    print_history()
    user_input = st.chat_input("질문을 입력하세요")
    if user_input:
        # ユーザーの入力を表示
        st.chat_message("user").write(user_input)

        # 選択されたカテゴリーに基づいてチェーンを作成
        chain = create_chain(selected_option)

        # AIの返答を格納する変数
        ai_answer = ""
        # AIの応答をストリームで受け取る
        with st.spinner("AIが考えています..."):
            response = chain.stream({"question": user_input})

        with st.chat_message("assistant"):
            container = st.empty()
            for chunk in response:
                ai_answer += chunk
                container.markdown(ai_answer)

        # 会話履歴にユーザーとAIのメッセージを追加（ChatMessage形式）
        add_history("user", user_input)
        add_history("assistant", ai_answer)


if __name__ == "__main__":
    main()
