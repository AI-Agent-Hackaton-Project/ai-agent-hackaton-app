from PIL import Image
from dotenv import load_dotenv
from vertexai.preview.vision_models import ImageGenerationModel
from langchain_google_vertexai import ChatVertexAI
from config.constants import JAPAN_PREFECTURES as PREFECTURES_LIST
import streamlit as st
import vertexai
import io
import os
import json

# .envファイルをロード
load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION")

MODEL_LOADED = False
LLM_LOADED = False

# Vertex AIの初期化
if PROJECT_ID and LOCATION:
    try:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        # 画像生成モデルをロード
        model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")
        MODEL_LOADED = True

        # LangChain LLMを初期化
        llm = ChatVertexAI(
            model_name="gemini-2.0-flash-lite-001",
            project=PROJECT_ID,
            location=LOCATION,
            temperature=0.1,
        )
        LLM_LOADED = True

        st.sidebar.success("Vertex AIと画像生成モデル、LLMの準備ができました。")
    except Exception as e:
        st.error(f"Vertex AIの初期化またはモデルのロードに失敗しました: {e}")
        MODEL_LOADED = False
        LLM_LOADED = False
else:
    st.error("環境変数 GCP_PROJECT_ID または GCP_LOCATION が設定されていません。")
    MODEL_LOADED = False
    LLM_LOADED = False

# セッション状態で都道府県データを管理
if "prefecture_data" not in st.session_state:
    st.session_state.prefecture_data = {}


def generate_single_prefecture_data(prefecture_name: str):
    """
    単一の都道府県のデータをLLMで生成する関数
    """
    if not LLM_LOADED:
        st.error("LLMが初期化されていません。")
        return None

    prompt = f"""以下の日本の都道府県について、その地域の魅力を伝えるための**4つの視覚的に印象的なシーン**を、画像生成用プロンプトとしてJSON形式で出力してください。

【目的】
画像生成AIが「1枚絵」として自然に描けるよう、以下の条件を厳密に守ってください。

【出力仕様】
- 各シーンは1文で完結し、1つの絵に描ける「情景」として成立する必要があります。
- その都道府県を象徴する「ランドマーク」「名物（料理や伝統工芸品）」「風景（季節感や自然）」「文化的象徴（伝統、祭り、色彩など）」を、**1文の中に自然に融合して描写**してください。
- その土地に詳しくない人でも視覚で理解できるよう、**誰もが知る代表的なもの**（例：福岡なら福岡タワー、太宰府天満宮、博多ラーメン、屋台文化 など）を必ず含めてください。
- 特に**その地域ならではの料理や食べ物（例：博多ラーメン、明太子、たこ焼き、ずんだ餅、きりたんぽなど）**を最低1つ以上、**必ずどこかのシーンに登場**させてください。
- 各文は日本語で書き、絵として描いたときに**情報過多にならず、統一感のある1枚**に仕上がるように注意してください。
- **文字（地名や人名、看板の文字など）や特定人物は描写しないでください。**

都道府県: {prefecture_name}

【出力形式】
以下のようなJSON形式で出力してください：

{{
  "{prefecture_name}": {{
    "prompts": [
      "ここに情景1の日本語プロンプトを記載（例: ○○神社の鳥居の前に、名物の○○が置かれた風景...）",
      "情景2",
      "情景3",
      "情景4"
    ],
    "theme": "{prefecture_name}の魅力を1枚ずつ凝縮した4つの視覚的情景"
  }}
}}
"""

    try:
        response = llm.invoke(prompt)
        response_text = (
            response.content if hasattr(response, "content") else str(response)
        )

        # JSON部分を抽出
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.rfind("```")
            json_text = response_text[start:end].strip()
        else:
            json_text = response_text.strip()

        generated_data = json.loads(json_text)
        return generated_data

    except json.JSONDecodeError as e:
        st.error(f"JSONパースエラー: {e}")
        st.error(f"LLMの回答: {response_text}")
        return None
    except Exception as e:
        st.error(f"データ生成エラー: {e}")
        return None


def generate_landscape_comic_prompt(prefecture_name: str) -> str:
    """
    選択された都道府県に基づき、風景・物だけの4コマ画像を生成するプロンプトを作成する関数
    """
    if prefecture_name not in st.session_state.prefecture_data:
        return f"Error: {prefecture_name}のデータが見つかりません。"

    data = st.session_state.prefecture_data[prefecture_name]

    prompt = f"""
※以下を厳守してください。

【1】画像内に**一切の文字を入れないでください。**  
- 日本語・英語を含むすべての言語の文字を禁止します。  
- 背景、看板、建物、商品、標識、ロゴ、装飾文字も禁止です。  
- 画像内に文字が一切ない状態で、{prefecture_name}の風景と物だけを描写してください。
- もし、文字が含まれている場合は、英語で人間が読めるようにしてください。

【2】白い枠・フチ・余白を**描かないでください。**  
- 各コマはキャンバスの端までしっかりと描写し、空白を作らないでください。

【3】全体のスタイル】  
- その地域がどこか分かるような風景画スタイルでしてください。  
- 配色は{prefecture_name}の地域の雰囲気に合うものにしてください。  
- 4コマ正方形レイアウトで、統一感のある美しい構成にしてください。
- それぞれ似てない内容で、かつ全体として一貫性のあるテーマを持たせてください。
- 6コマにはしないでください。各コマの指示を通りに4コマの正方形レイアウトでお願いします。
- 高画質なアニメ風の画像を生成してください。

【4】テーマ】  
- 全体のテーマは「{prefecture_name}の{data['theme']}を誰でも分かるような風景と物で表現する」

【5】各コマの指示】  

●1コマ目
{data["prompts"][0]}の内容で「ランドマーク」「名物（料理や伝統工芸品）」「風景（季節感や自然）」「文化的象徴（伝統、祭り、色彩など）」が分かるような画像にしてください。

●2コマ目  
{data["prompts"][1]}の内容を「ランドマーク」「名物（料理や伝統工芸品）」「風景（季節感や自然）」「文化的象徴（伝統、祭り、色彩など）」が分かるような画像にしてください。

●3コマ目
{data["prompts"][2]}の内容を「ランドマーク」「名物（料理や伝統工芸品）」「風景（季節感や自然）」「文化的象徴（伝統、祭り、色彩など）」が分かるような画像にしてください。

●4コマ目
{data["prompts"][3]}の内容を「ランドマーク」「名物（料理や伝統工芸品）」「風景（季節感や自然）」「文化的象徴（伝統、祭り、色彩など）」が分かるような画像にしてください。

この1~4コマは「{prefecture_name}の魅力を人物なしでリアルな風景と物だけで表現」することを目的としています。  
**文字なし・人物なしで、{prefecture_name}の美しさと特色を丁寧に描写してください。**


"""
    return prompt.strip()


def auto_generate_data_and_image(prefecture_name: str):
    """
    都道府県のデータ生成 → 画像生成を自動で連続実行する関数
    """
    success = False

    # ステップ1: データ生成
    if prefecture_name not in st.session_state.prefecture_data:
        st.info(f"📊 ステップ1: {prefecture_name}のデータを生成中...")
        generated_data = generate_single_prefecture_data(prefecture_name)

        if generated_data and prefecture_name in generated_data:
            st.session_state.prefecture_data[prefecture_name] = generated_data[
                prefecture_name
            ]
            st.success(f"✅ {prefecture_name}のデータ生成が完了しました！")
        else:
            st.error("❌ データの生成に失敗しました。")
            return False
    else:
        st.info(f"📊 {prefecture_name}のデータは既に存在します。")

    # ステップ2: 画像生成
    if prefecture_name in st.session_state.prefecture_data:
        data = st.session_state.prefecture_data[prefecture_name]
        st.info(f"🎨 ステップ2: {prefecture_name}の4コマ風景画像を生成中...")

        try:
            comic_prompt = generate_landscape_comic_prompt(prefecture_name)

            images = model.generate_images(
                prompt=comic_prompt,
                number_of_images=1,
                aspect_ratio="1:1",
                language="ja",
            )

            if images:
                pil_image = None
                if (
                    hasattr(images[0], "_pil_image")
                    and images[0]._pil_image is not None
                ):
                    pil_image = images[0]._pil_image.copy()
                elif hasattr(images[0], "load_image_bytes"):
                    try:
                        image_bytes = images[0].load_image_bytes()
                        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    except Exception as img_load_e:
                        st.error(f"画像変換エラー: {img_load_e}")

                if pil_image:
                    st.success(
                        f"✅ {prefecture_name}の4コマ風景画像生成が完了しました！"
                    )

                    # 結果表示
                    st.subheader("🖼️ 生成された4コマ風景画像")
                    st.info(f"テーマ: 「{prefecture_name}の{data['theme']}」")

                    st.image(
                        pil_image,
                        caption=f"{prefecture_name}の4コマ風景画像: 「{data['theme']}」",
                        use_container_width=True,
                    )

                    # 生成されたデータの詳細表示
                    with st.expander(f"📊 {prefecture_name}の生成データ詳細"):
                        col1, col2 = st.columns(2)
                        with col1:
                            for item in data["prompts"]:
                                st.write(f"• {item}")

                    # プロンプト詳細表示
                    with st.expander("📝 生成プロンプト詳細"):
                        st.text_area("プロンプト詳細", comic_prompt, height=400)

                    success = True
                else:
                    st.error("❌ 画像の表示に失敗しました。")
            else:
                st.error("❌ 画像の生成に失敗しました。")

        except Exception as e:
            st.error(f"❌ 画像生成エラー: {e}")

    return success


# メインアプリケーション
st.title("🗾 日本都道府県 風景4コマ画像ジェネレーター")
st.markdown(
    """
各都道府県の代表的な風景・名所・名物だけで構成される4コマ画像を自動生成します。
地域を選択して「自動生成開始」ボタンを押すと、データ生成→画像生成が連続で実行されます。
"""
)

# 都道府県選択
st.sidebar.header("📍 都道府県を選択")
selected_prefecture = st.sidebar.selectbox(
    "4コマ画像を生成したい都道府県を選択してください:",
    PREFECTURES_LIST,
    index=None,
    placeholder="都道府県を選んでください",
)

# 選択された都道府県がある場合のメイン処理
if selected_prefecture:
    st.sidebar.header("🎨 4コマ風景画像自動生成")

    # 自動生成開始ボタン
    if st.sidebar.button(
        f"🚀 {selected_prefecture}の4コマ画像を自動生成",
        type="primary",
        use_container_width=True,
    ):
        if MODEL_LOADED and LLM_LOADED:
            with st.spinner(f"{selected_prefecture}のデータ生成と画像生成を実行中..."):
                # データ生成 → 画像生成を自動実行
                auto_generate_data_and_image(selected_prefecture)
        else:
            st.error("❌ 画像生成モデルまたはLLMが初期化されていません。")

    # 既存データがある場合の情報表示
    if selected_prefecture in st.session_state.prefecture_data:
        st.info(
            "💡 この都道府県のデータは既に生成済みです。上のボタンを押すと画像生成のみ実行されます。"
        )

# サイドバーに現在の状況を表示
st.sidebar.header("📊 現在の状況")
st.sidebar.metric("生成済み都道府県数", len(st.session_state.prefecture_data))

if st.session_state.prefecture_data:
    st.sidebar.write("**生成済み都道府県:**")
    for pref in list(st.session_state.prefecture_data.keys()):
        st.sidebar.write(f"• {pref}")

# データリセット機能
if st.sidebar.button("🗑️ 全データをリセット", type="secondary"):
    st.session_state.prefecture_data = {}
    st.sidebar.success("全データがリセットされました。")
    st.rerun()
