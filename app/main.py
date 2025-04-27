import streamlit as st
from components.simple_chatbot import show_simple_chatbot


def main():
    """
    Streamlitアプリケーションのメイン処理を行う関数
    """
    st.title("シンプルなStreamlitアプリ")

    show_simple_chatbot()

    st.write("---")
    st.write("develop branch!")


if __name__ == "__main__":
    main()
