import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv


def show_simple_chatbot():
    load_dotenv()

    try:
        llm = ChatOpenAI(model="gpt-4o-mini")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたは親切で少しユーモラスなAIアシスタントです。ユーザーの質問に答えてください。",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser

    except Exception as e:
        st.error(f"LangChainの初期化中にエラーが発生しました: {e}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "こんにちは！何かお手伝いできることはありますか？",
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("メッセージを入力してください...")

    if user_input:
        # ユーザーメッセージ処理 (変更なし)
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        langchain_chat_history = []
        recent_messages = st.session_state.messages
        for msg in recent_messages[:-1]:
            if msg["role"] == "user":
                langchain_chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_chat_history.append(AIMessage(content=msg["content"]))

        try:
            with st.spinner("AIが考え中です..."):
                ai_response = chain.invoke(
                    {"input": user_input, "chat_history": langchain_chat_history}
                )
            st.session_state.messages.append(
                {"role": "assistant", "content": ai_response}
            )
            with st.chat_message("assistant"):
                st.markdown(ai_response)
        except Exception as e:
            st.error(f"AI応答の取得中にエラーが発生しました: {e}")

    with st.sidebar:
        st.header("情報")
        st.markdown(
            "これは Streamlit と LangChain を使ったシンプルなチャットボットのデモです。\n\nAPIキーは `.env` ファイルまたは Streamlit Secrets から読み込みます。"
        )
        if st.button("チャット履歴をクリア"):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "こんにちは！何かお手伝いできることはありますか？",
                }
            ]
            st.rerun()
