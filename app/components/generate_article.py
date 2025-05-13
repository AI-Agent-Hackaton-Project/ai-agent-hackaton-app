from vertexai.generative_models import GenerativeModel
import streamlit as st
from config.config import get_config
from langchain_google_vertexai import ChatVertexAI


def generate_article():
    config = get_config()

    llm = ChatVertexAI(
        model=config["model_name"],
        temperature=0,
        max_tokens=None,
        max_retries=6,
        stop=None,
    )
    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to French. Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]

    ai_msg = llm.invoke(messages)
    st.write(ai_msg.content)
