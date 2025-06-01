import streamlit as st
from components.map_section import map_section
from components.article_html_section import article_generator_app


def main():
    """
    Streamlitアプリケーションのメイン処理を行う関数
    """
    st.title("シンプルなStreamlitアプリ")

    map_section()

    selected_prefecture_name = st.session_state.get("selected_prefecture_info")

    if selected_prefecture_name:
        article_generator_app(selected_prefecture_name)

    st.write("---")


if __name__ == "__main__":
    main()
