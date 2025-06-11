from PIL import Image
from vertexai.preview.vision_models import ImageGenerationModel
from langchain_google_vertexai import ChatVertexAI
import io
import json
import tempfile

from config.env_config import get_env_config

MODEL_LOADED = False
LLM_LOADED = False
model = None
llm = None
config_settings = None

print("アプリケーション設定の読み込みを開始します...")
config_settings = get_env_config()

print(f"画像生成モデル ({config_settings['image_model_name']}) のロードを開始します...")
model = ImageGenerationModel.from_pretrained(config_settings["image_model_name"])
MODEL_LOADED = True
print(f"✅ 画像生成モデル ({config_settings['image_model_name']}) の準備ができました。")

print(f"LLM ({config_settings['model_name']}) の初期化を開始します...")
llm = ChatVertexAI(
    model_name=config_settings["model_name"],
    project=config_settings["gcp_project_id"],
    location=config_settings["gcp_location"],
    temperature=0.1,
)
LLM_LOADED = True
print(f"✅ LLM ({config_settings['model_name']}) の準備ができました。")
print("\n✨ 全てのモデルの準備が整いました。✨\n")


prefecture_data_store = {}


def generate_single_prefecture_data(prefecture_name: str):
    """
    単一の都道府県のデータをLLMで生成する関数
    """
    if not LLM_LOADED or not llm:
        print("LLMが初期化されていません。処理を中断します。")
        return None
    if not config_settings:
        print("設定がロードされていません。処理を中断します。")
        return None

    print(f"▶️  LLMを使用して「{prefecture_name}」のシーン記述を生成します...")
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

        # JSON部分を抽出 (Markdownコードブロック ```json ... ``` に対応)
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_text = response_text[start:end].strip()
        elif "```" in response_text:  # ```のみで囲まれている場合も考慮
            start = response_text.find("```") + 3
            end = response_text.rfind("```")
            json_text = response_text[start:end].strip()
        else:
            json_text = response_text.strip()

        generated_data = json.loads(json_text)
        print(f"   📄 「{prefecture_name}」のシーン記述をLLMから取得しました。")
        return generated_data

    except json.JSONDecodeError as e:
        print(f"   ❌ JSONパースエラー: {e}")
        print(f"   LLMからの未加工の回答: {response_text}")
        return None
    except Exception as e:
        print(f"   ❌ データ生成中の予期せぬエラー: {e}")
        return None


def generate_landscape_comic_prompt(prefecture_name: str) -> str | None:
    """
    選択された都道府県に基づき、風景・物だけの4コマ画像を生成するプロンプトを作成する関数
    各コマが明確に異なる背景と特色を持つように改善
    """
    if prefecture_name not in prefecture_data_store:
        print(f"   ⚠️ 「{prefecture_name}」のプロンプト用データが見つかりません。")
        return None

    data = prefecture_data_store[prefecture_name]
    print(f"   🎨 「{prefecture_name}」の4コマ画像用プロンプトを組み立てています...")

    # 各コマのスタイルと構図を明確に分離
    panel_styles = [
        "昼間の建築風景, 空と建物のコントラスト",
        "室内または近接撮影, 温かみのある食事風景",
        "屋外の自然風景, 広大な景色または季節感",
        "文化的な装飾または工芸品のクローズアップ",
    ]

    panel_compositions = [
        "遠景から中景への構図",
        "中景から近景への構図",
        "パノラマまたは広角構図",
        "アップまたは装飾的構図",
    ]

    prompt = f"""
Create a 4-panel square comic layout image representing {prefecture_name}, Japan. Each panel must have COMPLETELY DIFFERENT backgrounds, themes, and visual styles.

【CRITICAL REQUIREMENTS】
1. **NO TEXT ANYWHERE** - Absolutely no Japanese, English, or any written characters, signs, or logos
2. **NO WHITE BORDERS** - Each panel fills completely to the edge with no margins
3. **4 DISTINCT THEMES** - Each panel represents a different aspect of {prefecture_name}
4. **HIGH-QUALITY ANIME STYLE** with regional color palette matching {prefecture_name}

【PANEL SPECIFICATIONS】

🏛️ **TOP-LEFT PANEL (Landmark/Architecture)**
Theme: {data["prompts"][0]}
Visual Style: {panel_styles[0]}
Composition: {panel_compositions[0]}
Focus: Iconic buildings, temples, towers, or architectural landmarks of {prefecture_name}
Background: Clear sky, urban or historic setting

🍜 **TOP-RIGHT PANEL (Food/Cuisine)**  
Theme: {data["prompts"][1]}
Visual Style: {panel_styles[1]}
Composition: {panel_compositions[1]}
Focus: Famous local dishes, ingredients, or food culture of {prefecture_name}
Background: Restaurant interior, kitchen, or dining atmosphere

🌸 **BOTTOM-LEFT PANEL (Nature/Seasons)**
Theme: {data["prompts"][2]}
Visual Style: {panel_styles[2]} 
Composition: {panel_compositions[2]}
Focus: Natural landscapes, seasonal beauty, or climate features of {prefecture_name}
Background: Mountains, forests, rivers, or seasonal scenery

🎭 **BOTTOM-RIGHT PANEL (Culture/Tradition)**
Theme: {data["prompts"][3]}
Visual Style: {panel_styles[3]}
Composition: {panel_compositions[3]}  
Focus: Traditional crafts, festivals, cultural symbols, or regional colors of {prefecture_name}
Background: Cultural venues, traditional settings, or decorative environments

【VISUAL CONSISTENCY】
- Each panel uses colors that represent {prefecture_name}'s regional identity
- Lighting and atmosphere should vary between panels (day/indoor/outdoor/artistic)
- NO people or characters - only landscapes, objects, and scenery
- Each panel tells a different visual story about {prefecture_name}

Generate this as ONE seamless 4-panel square grid image showing the diverse charm of {prefecture_name} through four completely different visual perspectives.
"""

    print(f"   ✅ 「{prefecture_name}」の多様性重視画像生成プロンプト完成。")
    return prompt.strip()


def generate_four_images(prefecture_name: str) -> str | None:
    """
    都道府県のデータ生成 → 画像生成を自動で連続実行し、
    生成された画像を一時ファイルに保存してそのパスを返す関数。
    """
    print(f"\n🚀 「{prefecture_name}」の画像生成プロセスを開始します。")
    if not MODEL_LOADED or not LLM_LOADED or not model or not llm:
        print("❌ 画像生成モデルまたはLLMが初期化されていません。処理を中止します。")
        return None
    if not config_settings:
        print("❌ 設定情報がロードされていません。処理を中止します。")
        return None

    # ステップ1: データ生成 (プロンプトの元となるシーン記述)
    if prefecture_name not in prefecture_data_store:
        print(f"📊 ステップ1: 「{prefecture_name}」のシーン記述データを生成します。")
        generated_data = generate_single_prefecture_data(prefecture_name)

        if generated_data and prefecture_name in generated_data:
            prefecture_data_store[prefecture_name] = generated_data[prefecture_name]
            print(f"   ✅ 「{prefecture_name}」のシーン記述データ生成完了！")
        else:
            print(f"   ❌ 「{prefecture_name}」のシーン記述データ生成に失敗しました。")
            return None
    else:
        print(
            f"📊 ステップ1: 「{prefecture_name}」のシーン記述データはキャッシュに存在します。"
        )

    # ステップ2: 画像生成
    if prefecture_name in prefecture_data_store:
        print(f"🎨 ステップ2: 「{prefecture_name}」の4コマ風景画像を生成します。")

        try:
            comic_prompt = generate_landscape_comic_prompt(prefecture_name)
            if not comic_prompt:  # プロンプト生成に失敗した場合
                return None

            print(
                f"\n   📝 画像生成モデルへの最終プロンプト (一部):\n   {comic_prompt[:200]}...\n"
            )  # 長すぎるので一部表示

            images = model.generate_images(
                prompt=comic_prompt,
                number_of_images=1,
                aspect_ratio="1:1",
                language="ja",
            )

            if images:
                pil_image = None
                print("   🖼️ 画像データを処理中...")
                # Vertex AI SDK の Image オブジェクトからPillow Imageへの変換
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
                        print(
                            f"   ❌ 画像バイトデータのPillowイメージへの変換エラー: {img_load_e}"
                        )
                        return None

                if pil_image:
                    print("   💾 生成画像を一時ファイルに保存中...")
                    # ファイル名を安全にするため、英数字以外をアンダースコアに置換
                    safe_prefecture_name = "".join(
                        c if c.isalnum() else "_" for c in prefecture_name
                    )

                    # 一時ファイルを作成 (自動削除はしないので、後でパスを利用可能)
                    temp_file = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".png",
                        prefix=f"{safe_prefecture_name}_4koma_",
                    )
                    image_path = temp_file.name
                    pil_image.save(image_path)
                    temp_file.close()  # ファイルハンドルを閉じる

                    print(f"   ✅ 「{prefecture_name}」の4コマ風景画像生成完了！")
                    print(f"   📍 一時保存先: {image_path}")
                    return image_path
                else:
                    print("   ❌ 画像のPillow Imageオブジェクトの取得に失敗しました。")
                    return None
            else:
                print("   ❌ 画像生成モデルから画像が返されませんでした。")
                return None

        except Exception as e:
            print(f"   ❌ 画像生成中の予期せぬエラー: {e}")
            return None
    else:
        print(
            f"   ❌ 「{prefecture_name}」のデータが見つからないため画像生成をスキップします。"
        )
        return None
