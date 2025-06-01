import streamlit as st
import os
from dotenv import load_dotenv
import vertexai
import base64
from io import BytesIO

from langchain_google_vertexai import ChatVertexAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

from vertexai.preview.vision_models import ImageGenerationModel
from typing import Optional
import traceback

try:
    from utils.generate_titles import generate_titles_for_prefecture
    from config.constants import JAPAN_PREFECTURES
except ImportError as e:
    st.error(f"必須モジュールのインポートに失敗しました: {e}")
    st.error(
        "utils.generate_titlesとconfig.constantsファイルがプロジェクトに含まれているか確認してください。"
    )
    st.stop()

load_dotenv()
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION")

if not GCP_PROJECT_ID or not GCP_LOCATION:
    st.error("GCP_PROJECT_IDまたはGCP_LOCATION環境変数が設定されていません。")
    st.stop()

IMAGE_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=[
        "prefecture",
        "main_title",
        "sub_title",
        "regional_characteristics",
    ],
    template="""
以下の情報を基に、画像生成AI用の詳細なプロンプトを作成してください。

地域: {prefecture}
メインテーマ: {main_title}
サブテーマ: {sub_title}
地域特性: {regional_characteristics}

要求:
1. {prefecture}の特徴的な風景や建物、文化を含める
2. {main_title}と{sub_title}のテーマを視覚的に表現する
3. 日本のアニメ風の美しい背景画として描く
4. 4:3のアスペクト比で制作
5. 文字やテキストは一切含めない

出力は画像生成AIに直接入力できる形式の英語プロンプトで記述してください。
""",
)


@st.cache_resource
def initialize_vertex_ai():
    """Vertex AI初期化"""
    try:
        st.info(
            f"Vertex AI初期化中... プロジェクト: {GCP_PROJECT_ID}, 場所: {GCP_LOCATION}"
        )
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        image_model = ImageGenerationModel.from_pretrained(
            "imagen-3.0-fast-generate-001"
        )
        llm = ChatVertexAI(
            model_name="gemini-2.0-flash-lite-001",
            project=GCP_PROJECT_ID,
            location=GCP_LOCATION,
            temperature=0.7,
        )
        st.success("Vertex AIとLLMの初期化に成功しました!")
        return image_model, llm
    except Exception as e:
        st.error(f"Vertex AI初期化失敗: {e}")
        st.code(traceback.format_exc())
        return None, None


@st.cache_data(show_spinner=False)
def generate_regional_characteristics(_llm: ChatVertexAI, prefecture: str) -> str:
    """指定された都道府県の地域特性をAIで生成"""
    prompt = f"""
{prefecture}の代表的な特徴を教えてください。以下の要素を含めて150文字程度で記述してください：

1. 有名な観光地やランドマーク
2. 自然環境の特徴
3. 文化的な特色
4. 特産品や名物

例：沖縄県の場合
美しい海とサンゴ礁、首里城などの琉球文化が色濃い沖縄県。

・観光地・ランドマーク: 美ら海水族館、首里城、国際通り
・自然環境: エメラルドグリーンの海、亜熱帯気候、多様な動植物
・文化: 琉球文化、エイサー、三線
・特産品・名物: 海ぶどう、ゴーヤチャンプルー、泡盛

{prefecture}について同様の形式で回答してください。
"""
    try:
        with st.spinner(f"{prefecture}の地域特性を生成中..."):
            response = _llm.invoke([HumanMessage(content=prompt)])
            result = response.content.strip()
            return result if result else f"{prefecture}の美しい自然と文化的特徴"
    except Exception as e:
        st.warning(f"地域特性生成エラー: {e}")
        return f"{prefecture}の美しい自然と伝統的な文化"


def generate_image_prompt(
    llm: ChatVertexAI,
    prefecture: str,
    main_title: str,
    sub_title: str,
    regional_chars: str,
) -> str:
    """画像生成用のプロンプトを生成"""
    try:
        prompt_input = IMAGE_PROMPT_TEMPLATE.format(
            prefecture=prefecture,
            main_title=main_title,
            sub_title=sub_title,
            regional_characteristics=regional_chars,
        )
        with st.spinner(
            "画像生成プロンプトを作成中... (これはUIには表示されません)"
        ):  # スピナーメッセージを調整
            response = llm.invoke([HumanMessage(content=prompt_input)])
            generated_prompt = response.content.strip()
        if not generated_prompt:
            return f"Beautiful anime-style landscape of {prefecture}, Japan, featuring {regional_chars}, with theme of {sub_title}, 4:3 aspect ratio, high quality, detailed background art, no text, no characters"
        return generated_prompt
    except Exception as e:
        st.error(f"プロンプト生成エラー: {e}")
        return f"Beautiful anime-style landscape of {prefecture}, Japan, featuring {regional_chars}, with theme of {sub_title}, 4:3 aspect ratio, high quality, detailed background art, no text, no characters"


def generate_image_with_model(
    image_model: ImageGenerationModel, prompt: str
) -> Optional[bytes]:
    """画像生成モデルで画像を生成"""
    try:
        negative_prompt = "text, words, letters, signs, logos, watermarks, signature, ugly, low quality, blurry, deformed, malformed, extra limbs, bad art, faces, people, characters, nsfw"
        with st.spinner("画像を生成中... (少し時間がかかります)"):
            response = image_model.generate_images(
                prompt=prompt,
                number_of_images=1,
                aspect_ratio="4:3",
                negative_prompt=negative_prompt,
            )
        if response and response.images:
            image_obj = response.images[0]
            if hasattr(image_obj, "_image_bytes") and image_obj._image_bytes:
                return image_obj._image_bytes
            elif hasattr(image_obj, "image_bytes") and image_obj.image_bytes:
                return image_obj.image_bytes
            elif hasattr(image_obj, "_pil_image") and image_obj._pil_image:
                buffered = BytesIO()
                image_obj._pil_image.save(buffered, format="PNG")
                return buffered.getvalue()
        return None
    except Exception as e:
        st.error(f"画像生成エラー: {e}")
        st.code(traceback.format_exc())
        return None


def image_to_base64(image_bytes: bytes) -> str:
    """画像バイトをbase64文字列に変換"""
    return base64.b64encode(image_bytes).decode("utf-8")


st.set_page_config(page_title="日本地域画像生成アプリ", layout="wide")
st.title("🗾 日本地域特性画像生成アプリ")
st.markdown("選択した都道府県の特性を活かした美しいアニメ風画像を5枚生成します。")

if not JAPAN_PREFECTURES:
    st.error("都道府県リストを読み込めません。")
    st.stop()

image_model, llm = initialize_vertex_ai()

if not image_model or not llm:
    st.error("モデルの初期化に失敗しました。")
    st.stop()

with st.sidebar:
    st.header("⚙️ 設定")
    selected_prefecture = st.selectbox("📍 都道府県を選択", JAPAN_PREFECTURES)
    regional_characteristics = ""
    if selected_prefecture:
        st.subheader(f"📍 {selected_prefecture} の特性")
        regional_characteristics = generate_regional_characteristics(
            llm, selected_prefecture
        )
        st.info(regional_characteristics)
    st.markdown("---")
    st.subheader("📋 生成結果管理")
    if st.button("🗑️ 生成画像をクリア"):
        if "article_images" in st.session_state:
            del st.session_state.article_images
            st.success("生成された画像データをクリアしました。")
        else:
            st.info("クリアする画像データがありません。")
        st.rerun()

st.subheader("🗾 選択された地域")
if selected_prefecture:
    st.info(f"**{selected_prefecture}**")
else:
    st.info("都道府県を選択してください。")

if st.button(
    "🎨 画像生成開始",
    type="primary",
    use_container_width=True,
    disabled=not selected_prefecture,
):
    if not selected_prefecture:
        st.warning("まず都道府県を選択してください。")
    else:
        progress_bar = st.progress(0, text="テーマ生成中...")
        try:
            result = generate_titles_for_prefecture(selected_prefecture)
            titles_output = result.get("titles_output")
            if not titles_output:
                st.error("テーマ生成に失敗しました。")
                st.stop()
            main_title = titles_output.get("main_title", "地域の美")
            sub_titles_from_api = titles_output.get("sub_titles", ["伝統的な風景"])
            progress_bar.progress(20, text="テーマ生成完了")
        except Exception as e:
            st.error(f"テーマ生成エラー: {e}")
            st.code(traceback.format_exc())
            st.stop()

        num_target_images = 5
        final_sub_titles = list(sub_titles_from_api)
        if len(final_sub_titles) < num_target_images:
            for i in range(len(final_sub_titles), num_target_images):
                final_sub_titles.append(f"{main_title} - 生成テーマ {i + 1}")
        elif len(final_sub_titles) > num_target_images:
            final_sub_titles = final_sub_titles[:num_target_images]

        st.subheader("📖 生成されたテーマ")
        st.markdown(f"**メインテーマ:** {main_title}")
        st.markdown("**サブテーマ (画像生成に使用):**")
        for i, subtitle in enumerate(final_sub_titles, 1):
            st.markdown(f"{i}. {subtitle}")

        st.markdown("---")
        st.subheader(f"🖼️ 生成された画像 ({selected_prefecture})")

        article_images_temp_list = []

        if "article_images" in st.session_state:
            del st.session_state.article_images

        cols = st.columns(num_target_images)

        for i, subtitle in enumerate(final_sub_titles):
            current_progress = 30 + (i * 70 // num_target_images)
            progress_bar.progress(
                current_progress,
                text=f"画像 {i+1}/{num_target_images} 生成中 ('{subtitle}')...",
            )

            image_prompt = generate_image_prompt(
                llm, selected_prefecture, main_title, subtitle, regional_characteristics
            )

            image_bytes = generate_image_with_model(
                image_model,
                image_prompt,
            )

            if image_bytes:
                st.image(
                    image_bytes,
                    caption=f"{i+1}. {subtitle}",
                    use_container_width=True,
                )
                base64_data = image_to_base64(image_bytes)
                image_info = {
                    "src": f"data:image/png;base64,{base64_data}",
                    "alt": subtitle,
                }
                article_images_temp_list.append(image_info)
            else:
                st.error(f"画像 {i+1} ('{subtitle}') の生成に失敗しました。")

        progress_bar.progress(100, text="画像生成完了!")

        st.session_state.article_images = article_images_temp_list  # フィルタリング不要

        if st.session_state.article_images:
            st.success(
                f"✅ {len(st.session_state.article_images)} 枚の画像を生成し、`st.session_state.article_images` に保存しました。"
            )
            st.caption(
                "これらの画像は、他のページやアプリケーションの別箇所で参照可能です。"
            )
        else:  # article_images_temp_list が空の場合（すべての画像生成に失敗）
            st.error("画像の生成にすべて失敗しました。")

        if "article_images" in st.session_state and st.session_state.article_images:
            st.markdown("---")
            st.subheader("🖼️ 現在の生成画像データ")
            st.caption(
                f"{len(st.session_state.article_images)} 件の画像データが `st.session_state` にあります。"
            )
            st.json(st.session_state.article_images)  # st.json で整形して表示
