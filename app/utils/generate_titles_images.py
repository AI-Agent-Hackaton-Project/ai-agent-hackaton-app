import os
import tempfile
import traceback
from io import BytesIO
from typing import List, Dict, Optional, Tuple

import vertexai
from langchain_google_vertexai import ChatVertexAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from vertexai.preview.vision_models import ImageGenerationModel

from dotenv import load_dotenv

load_dotenv()


# 元のコードから流用するプロンプトテンプレート
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


def _initialize_vertex_ai_core(
    project_id: str,
    location: str,
    image_gen_model_name: str = "imagen-3.0-fast-generate-001",
    llm_model_name: str = "gemini-2.0-flash-lite-001",  # 元のコードでは gemini-1.0-pro-002 だったが、新しいモデルに変更も可能
) -> Tuple[Optional[ImageGenerationModel], Optional[ChatVertexAI]]:
    """Vertex AI（画像生成モデルとLLM）を初期化します。 (Streamlit非依存)"""
    try:
        print(f"Vertex AI初期化中... プロジェクト: {project_id}, 場所: {location}")
        vertexai.init(project=project_id, location=location)
        image_model = ImageGenerationModel.from_pretrained(image_gen_model_name)
        llm = ChatVertexAI(
            model_name=llm_model_name,
            project=project_id,
            location=location,
            temperature=0.7,
        )
        print("Vertex AIとLLMの初期化に成功しました!")
        return image_model, llm
    except Exception as e:
        print(f"Vertex AI初期化失敗: {e}")
        print(traceback.format_exc())
        return None, None


def _generate_regional_characteristics_core(llm: ChatVertexAI, prefecture: str) -> str:
    """指定された都道府県の地域特性をAIで生成します。 (Streamlit非依存)"""
    prompt_text = f"""
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
        print(f"{prefecture}の地域特性を生成中...")
        response = llm.invoke([HumanMessage(content=prompt_text)])
        result = response.content.strip()
        if result:
            print(f"地域特性生成成功: {result[:50]}...")
            return result
        else:
            print(
                f"{prefecture}の地域特性生成で空の結果が返されました。デフォルト値を使用します。"
            )
            return f"{prefecture}の美しい自然と文化的特徴"
    except Exception as e:
        print(f"地域特性生成エラー: {e}")
        return f"{prefecture}の美しい自然と伝統的な文化"


def _generate_image_prompt_core(
    llm: ChatVertexAI,
    prefecture: str,
    main_title: str,
    sub_title: str,
    regional_chars: str,
) -> str:
    """画像生成用のプロンプトをLLMで生成します。 (Streamlit非依存)"""
    try:
        prompt_input = IMAGE_PROMPT_TEMPLATE.format(
            prefecture=prefecture,
            main_title=main_title,
            sub_title=sub_title,
            regional_characteristics=regional_chars,
        )
        print(f"「{sub_title}」の画像生成プロンプトを作成中...")
        response = llm.invoke([HumanMessage(content=prompt_input)])
        generated_prompt = response.content.strip()
        if generated_prompt:
            print(f"  プロンプト生成成功: {generated_prompt[:50]}...")
            return generated_prompt
        else:
            print(
                "  画像生成プロンプトが空でした。フォールバックプロンプトを使用します。"
            )
            return f"Beautiful anime-style landscape of {prefecture}, Japan, featuring {regional_chars}, with theme of {sub_title}, 4:3 aspect ratio, high quality, detailed background art, no text, no characters"
    except Exception as e:
        print(f"  プロンプト生成エラー: {e}")
        return f"Beautiful anime-style landscape of {prefecture}, Japan, featuring {regional_chars}, with theme of {sub_title}, 4:3 aspect ratio, high quality, detailed background art, no text, no characters"


def _generate_image_with_model_core(
    image_model: ImageGenerationModel, prompt: str
) -> Optional[bytes]:
    """画像生成モデルで画像を生成し、バイトデータを返します。 (Streamlit非依存)"""
    try:
        negative_prompt = "text, words, letters, signs, logos, watermarks, signature, ugly, low quality, blurry, deformed, malformed, extra limbs, bad art, faces, people, characters, nsfw"
        print(f"  画像生成中 (プロンプト: {prompt[:30]}...)..")
        response = image_model.generate_images(
            prompt=prompt,
            number_of_images=1,
            aspect_ratio="4:3",  # 元のコードでは "16:9" だったが、IMAGE_PROMPT_TEMPLATE の要求は 4:3
            negative_prompt=negative_prompt,
            # 必要に応じて他のパラメータ (例: seed, guidance_scale) を追加
        )
        if response and response.images:
            image_obj = response.images[0]
            # Vertex AI SDKのバージョンによって属性名が異なる場合があるため、複数の可能性をチェック
            if hasattr(image_obj, "_image_bytes") and image_obj._image_bytes:
                print("  画像生成成功 (_image_bytes).")
                return image_obj._image_bytes
            elif (
                hasattr(image_obj, "image_bytes") and image_obj.image_bytes
            ):  # 新しいSDKでの標準的な属性
                print("  画像生成成功 (image_bytes).")
                return image_obj.image_bytes
            elif (
                hasattr(image_obj, "_pil_image") and image_obj._pil_image
            ):  # PILイメージオブジェクトの場合
                print("  画像生成成功 (_pil_image), PNGに変換中...")
                buffered = BytesIO()
                image_obj._pil_image.save(buffered, format="PNG")
                return buffered.getvalue()
            else:
                print("  画像データが見つかりませんでした。")
                return None
        print("  画像生成APIからの応答が不正です。")
        return None
    except Exception as e:
        print(f"  画像生成エラー: {e}")
        print(traceback.format_exc())
        return None


def generate_prefecture_image_and_get_path(
    prefecture: str,
    main_title: str,
    sub_titles: List[str],
    gcp_project_id: str,
    gcp_location: str,
    llm_model_name: str = "gemini-2.0-flash-lite-001",
    image_gen_model_name: str = "imagen-3.0-fast-generate-001",
) -> List[str]:
    """
    指定された都道府県、メインテーマ、サブテーマリストに基づき、
    複数の画像を生成し、一時ファイルとして保存後、そのパスのリストを返します。

    Args:
        prefecture (str): 都道府県名 (例: "東京都", "京都府")
        main_title (str): 画像生成の全体的なメインテーマ
        sub_titles (List[str]): 各画像に対応するサブテーマのリスト
        gcp_project_id (str): Google Cloud Project ID
        gcp_location (str): Google Cloud Location (例: "asia-northeast1")
        llm_model_name (str): 使用するLLMのモデル名
        image_gen_model_name (str): 使用する画像生成モデルのモデル名

    Returns:
        List[str]: 生成された各画像の一時ファイルパスのリスト。
                エラーが発生した場合や画像が生成できなかった場合は空のリストを返すこともあります。
                呼び出し側は、返されたパスのファイルを処理後、不要であれば削除する責任を持ちます。
    """
    if not all([prefecture, main_title, sub_titles, gcp_project_id, gcp_location]):
        print("エラー: 必須引数が不足しています。")
        return []

    generated_image_paths: List[str] = []
    temp_dir = tempfile.mkdtemp(prefix=f"generated_images_{prefecture}_")
    print(f"一時ディレクトリを作成しました: {temp_dir}")

    image_model, llm = _initialize_vertex_ai_core(
        gcp_project_id, gcp_location, image_gen_model_name, llm_model_name
    )

    if not image_model or not llm:
        print("モデルの初期化に失敗したため、処理を中断します。")
        return []

    regional_characteristics = _generate_regional_characteristics_core(llm, prefecture)
    if not regional_characteristics:
        regional_characteristics = f"{prefecture}の多様な魅力"

    for i, sub_title in enumerate(sub_titles):
        print(f"\n--- 画像 {i + 1}/{len(sub_titles)} ({sub_title}) の処理開始 ---")
        image_prompt = _generate_image_prompt_core(
            llm, prefecture, main_title, sub_title, regional_characteristics
        )

        image_bytes = _generate_image_with_model_core(image_model, image_prompt)

        if image_bytes:
            try:
                # ファイル名をサニタイズ（OSが許容しない文字を除去または置換）
                safe_subtitle = "".join(
                    c if c.isalnum() or c in (" ", "-", "_") else "_" for c in sub_title
                ).rstrip()
                safe_subtitle = safe_subtitle.replace(" ", "_")
                if len(safe_subtitle) > 50:  # ファイル名が長くなりすぎないように
                    safe_subtitle = safe_subtitle[:50]

                file_name = f"image_{i+1}_{safe_subtitle}.png"
                image_file_path = os.path.join(temp_dir, file_name)

                with open(image_file_path, "wb") as f:
                    f.write(image_bytes)
                generated_image_paths.append(image_file_path)
                print(f"  画像 {i + 1} を {image_file_path} に保存しました。")
            except Exception as e:
                print(f"  画像 {i + 1} のファイル保存中にエラーが発生しました: {e}")
                print(traceback.format_exc())
        else:
            print(
                f"  画像 {i + 1} ('{sub_title}') の生成に失敗しました。スキップします。"
            )

    if not generated_image_paths:
        print("すべての画像の生成に失敗したか、サブテーマが空でした。")

    print(f"\n--- 画像生成処理完了 ---")
    print(f"生成された画像パス (計 {len(generated_image_paths)} 枚):")
    for path in generated_image_paths:
        print(path)
    print(f"画像は一時ディレクトリ {temp_dir} に保存されています。")
    print("これらのファイルは、使用後に呼び出し側で削除する必要があります。")

    return generated_image_paths
