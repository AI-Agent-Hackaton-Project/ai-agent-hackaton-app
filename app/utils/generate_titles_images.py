import os
import tempfile
import traceback
from typing import List, Optional, Tuple

import vertexai
from langchain_google_vertexai import ChatVertexAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from vertexai.preview.vision_models import ImageGenerationModel

from dotenv import load_dotenv

# .envファイルから環境変数をロード
load_dotenv()


# 画像プロンプト生成用テンプレート - サブタイトル内容重視
SUBTITLE_FOCUSED_IMAGE_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=[
        "prefecture",
        "main_title",
        "sub_title",
        "regional_characteristics",
        "image_index",
        "total_images",
    ],
    template="""
あなたは創造的なアートディレクターです。以下の情報を基に、「{sub_title}」というサブタイトルの内容を{prefecture}の地域特色で表現する、魅力的な英語の画像生成プロンプトを作成してください。

# 基本情報
- 地域: {prefecture}
- 記事のメインテーマ: {main_title}
- **この画像の中心テーマ**: {sub_title}
- 地域の特徴: {regional_characteristics}
- 画像番号: {image_index}/{total_images}

# サブタイトル「{sub_title}」の表現要件
この画像は「{sub_title}」というテーマを{prefecture}らしさで表現する必要があります：

**内容分析**: 「{sub_title}」が何について語っているかを理解し、それを視覚的に表現してください
- もし歴史について → {prefecture}の歴史的建造物や文化遺産を中心に
- もし自然について → {prefecture}の特徴的な自然景観を中心に  
- もし食文化について → {prefecture}の代表的な料理や食材を中心に
- もし伝統について → {prefecture}の伝統工芸や文化的要素を中心に
- もし観光について → {prefecture}の有名観光スポットを中心に
- もし季節について → {prefecture}のその季節の特色を中心に
- もし産業について → {prefecture}の特色ある産業や技術を中心に

# 画像の多様性確保（{image_index}番目の画像として）
他の画像との差別化のため、以下の視覚的要素を適用：

{image_index}番目の画像の視覚的特徴:
- カメラアングル: {self._get_camera_angle(image_index)}
- 時間帯・照明: {self._get_lighting_condition(image_index)}
- 構図スタイル: {self._get_composition_style(image_index)}
- アートスタイル: {self._get_art_style(image_index)}
- 色彩傾向: {self._get_color_palette(image_index)}

# スタイル指定
- 基本スタイル: 美しいアニメの背景美術 (Beautiful anime background art)
- 品質: 高品質、詳細な描写 (highly detailed, high quality)
- 雰囲気: 「{sub_title}」のテーマに適した感情的な雰囲気

# 必須要素
- **「{sub_title}」の内容を{prefecture}の地域特色で具体的に表現**
- {prefecture}らしさが一目で分かる象徴的要素を含める
- 「{sub_title}」のテーマに最も関連する視覚的要素を中心に配置
- 文字、テキスト、人物の顔は含めない (no text, no letters, no character faces, no people)
- アスペクト比: 4:3

# 出力
「{sub_title}」というテーマを{prefecture}の地域特色で表現する英語のプロンプトのみを出力してください。説明やタイトルは不要です。
""",
)


def _get_camera_angle(image_index: int) -> str:
    """画像番号に基づいてカメラアングルを返す"""
    angles = [
        "遠景・俯瞰視点 (wide shot, aerial view)",
        "中景・人の目線 (medium shot, eye level)",
        "近景・ローアングル (close-up, low angle)",
        "クローズアップ・ハイアングル (extreme close-up, high angle)",
        "斜め視点・ダイナミック (diagonal perspective, dynamic angle)",
    ]
    # 画像番号を基にリストから選択（循環的に使用）
    return angles[(image_index - 1) % len(angles)]


def _get_lighting_condition(image_index: int) -> str:
    """画像番号に基づいて照明条件を返す"""
    lighting = [
        "朝の柔らかい光 (soft morning light, dawn)",
        "明るい昼光 (bright daylight, midday sun)",
        "夕焼けの温かい光 (warm sunset light, golden hour)",
        "夜景・ライトアップ (night scene, illuminated)",
        "曇り空・柔らかい拡散光 (overcast, soft diffused light)",
    ]
    # 画像番号を基にリストから選択（循環的に使用）
    return lighting[(image_index - 1) % len(lighting)]


def _get_composition_style(image_index: int) -> str:
    """画像番号に基づいて構図スタイルを返す"""
    compositions = [
        "パノラマ構図 (panoramic composition)",
        "中央集中構図 (centered composition)",
        "三分割構図 (rule of thirds)",
        "対角線構図 (diagonal composition)",
        "フレーミング構図 (framing composition)",
    ]
    # 画像番号を基にリストから選択（循環的に使用）
    return compositions[(image_index - 1) % len(compositions)]


def _get_art_style(image_index: int) -> str:
    """画像番号に基づいてアートスタイルを返す"""
    styles = [
        "リアリスティック風景画風 (realistic landscape painting)",
        "温かみのあるイラスト風 (warm illustration style)",
        "フォトリアリスティック (photorealistic style)",
        "シネマティック・ドラマチック (cinematic, dramatic style)",
        "アーティスティック・絵画風 (artistic, painterly style)",
    ]
    # 画像番号を基にリストから選択（循環的に使用）
    return styles[(image_index - 1) % len(styles)]


def _get_color_palette(image_index: int) -> str:
    """画像番号に基づいて色彩パレットを返す"""
    palettes = [
        "爽やかな青と緑 (fresh blue and green palette)",
        "温かいオレンジと茶色 (warm orange and brown palette)",
        "豊かな自然色彩 (rich natural earth tones)",
        "深い紫と金色 (deep purple and gold palette)",
        "季節感のある色彩 (seasonal color palette)",
    ]
    # 画像番号を基にリストから選択（循環的に使用）
    return palettes[(image_index - 1) % len(palettes)]


def _initialize_vertex_ai(
    project_id: str, location: str, image_gen_model_name: str, llm_model_name: str
) -> Tuple[Optional[ImageGenerationModel], Optional[ChatVertexAI]]:
    """Vertex AIモデルを初期化する"""
    try:
        print(f"🔧 Vertex AI初期化中... プロジェクト: {project_id}, 場所: {location}")
        # Vertex AIを初期化
        vertexai.init(project=project_id, location=location)
        # 画像生成モデルをロード
        image_model = ImageGenerationModel.from_pretrained(image_gen_model_name)
        # Chat LLMモデルを初期化
        llm = ChatVertexAI(
            model_name=llm_model_name,
            project=project_id,
            location=location,
            temperature=0.8,  # 創造性向上のため高く設定
        )
        print("✅ Vertex AIとLLMの初期化に成功しました!")
        return image_model, llm
    except Exception as e:
        print(f"❌ Vertex AI初期化失敗: {e}")
        # エラーのトレースバックを出力
        traceback.print_exc()
        return None, None


def _generate_regional_characteristics(llm: ChatVertexAI, prefecture: str) -> str:
    """地域の特性を生成する"""
    prompt_text = f"""
    {prefecture}について、画像生成に役立つ視覚的特徴を教えてください：

    【代表的観光地・建築物】
    - 有名な建物、神社、城、橋、タワーなどの具体的な形状・色彩・特徴
    
    【自然・地理的特徴】  
    - 特徴的な山、海、川、湖、平野、気候の視覚的特徴
    - 季節ごとの代表的な自然現象や植物
    
    【食文化・特産品】
    - 代表的な料理、食材、お土産の色彩・形状・盛り付け
    
    【文化・伝統】
    - 伝統工芸品、祭り、文化的象徴の視覚的特徴
    
    【都市景観・雰囲気】
    - 街並み、建物、道路の特徴、地域特有の雰囲気
    
    画像生成AIが理解できるよう、**具体的な色彩、形状、質感**を中心に記述してください。
    
    {prefecture}の視覚的特徴：
    """

    try:
        print(f"🌍 {prefecture}の地域特性を生成中...")
        # LLMを呼び出して地域特性を生成
        response = llm.invoke([HumanMessage(content=prompt_text)])
        characteristics = response.content.strip()
        print(f"📝 地域特性生成完了: {characteristics[:100]}...")
        # 生成された特性が空の場合はデフォルト値を返す
        return characteristics or f"{prefecture}の美しい自然と文化的特徴"
    except Exception as e:
        print(f"⚠️ 地域特性生成エラー: {e}")
        # エラー発生時のデフォルト値を返す
        return f"{prefecture}の美しい自然と伝統的な文化"


def _generate_image_prompt(
    llm: ChatVertexAI,
    prefecture: str,
    main_title: str,
    sub_title: str,
    regional_chars: str,
    image_index: int = 1,
    total_images: int = 1,
) -> str:
    """サブタイトルの内容を重視した画像生成用プロンプトを作成する"""
    try:
        # prefectureの値の検証とログ出力
        print(f"🏮 Prefecture 値の確認: '{prefecture}' (型: {type(prefecture)})")
        print(f"📝 サブタイトル: '{sub_title}' (型: {type(sub_title)})")
        print(f"🔢 画像インデックス: {image_index}/{total_images}")

        # prefectureが実際の地域名であるか確認し、無効な場合はデフォルトを設定
        if (
            not prefecture
            or not isinstance(prefecture, str)
            or len(prefecture.strip()) == 0
        ):
            print(
                f"⚠️ 無効なprefecture値: '{prefecture}'. デフォルト値 '日本' を使用します"
            )
            prefecture = "日本"

        # prefectureから不要な空白や特殊文字を削除
        prefecture_clean = prefecture.strip()
        print(f"✅ クリーンアップされたPrefecture: '{prefecture_clean}'")

        # ヘルパー関数を使用して多様な視覚的要素を生成
        camera_angle = _get_camera_angle(image_index)
        lighting = _get_lighting_condition(image_index)
        composition = _get_composition_style(image_index)
        art_style = _get_art_style(image_index)
        color_palette = _get_color_palette(image_index)

        # テンプレートにヘルパー関数の結果を含めてプロンプトを作成
        enhanced_template = f"""
あなたは創造的なアートディレクターです。以下の情報を基に、「{sub_title}」というサブタイトルの内容を{prefecture_clean}の地域特色で表現する、魅力的な英語の画像生成プロンプトを作成してください。

# 基本情報
- 地域: {prefecture_clean}
- 記事のメインテーマ: {main_title}
- **この画像の中心テーマ**: {sub_title}
- 地域の特徴: {regional_chars}
- 画像番号: {image_index}/{total_images}

# サブタイトル「{sub_title}」の表現要件
この画像は「{sub_title}」というテーマを{prefecture}らしさで表現する必要があります：

**内容分析**: 「{sub_title}」が何について語っているかを理解し、それを視覚的に表現してください
- もし歴史について → {prefecture}の歴史的建造物や文化遺産を中心に
- もし自然について → {prefecture}の特徴的な自然景観を中心に  
- もし食文化について → {prefecture}の代表的な料理や食材を中心に
- もし伝統について → {prefecture}の伝統工芸や文化的要素を中心に
- もし観光について → {prefecture}の有名観光スポットを中心に
- もし季節について → {prefecture}のその季節の特色を中心に
- もし産業について → {prefecture}の特色ある産業や技術を中心に

# 画像の多様性確保（{image_index}番目の画像として）
他の画像との差別化のため、以下の視覚的要素を適用：

{image_index}番目の画像の視覚的特徴:
- カメラアングル: {camera_angle}
- 時間帯・照明: {lighting}
- 構図スタイル: {composition}
- アートスタイル: {art_style}
- 色彩傾向: {color_palette}

# スタイル指定
- 基本スタイル: 美しいアニメの背景美術 (Beautiful anime background art)
- 品質: 高品質、詳細な描写 (highly detailed, high quality)
- 雰囲気: 「{sub_title}」のテーマに適した感情的な雰囲気

# 必須要素
- **「{sub_title}」の内容を{prefecture}の地域特色で具体的に表現**
- {prefecture}らしさが一目で分かる象徴的要素を含める
- 「{sub_title}」のテーマに最も関連する視覚的要素を中心に配置
- 文字、テキスト、人物の顔は含めない (no text, no letters, no character faces, no people)
- アスペクト比: 4:3

# 出力
「{sub_title}」というテーマを{prefecture}の地域特色で表現する英語のプロンプトのみを出力してください。説明やタイトルは不要です。
"""

        print(
            f"🎨 画像{image_index}「{sub_title}」のサブタイトル重視プロンプトを作成中..."
        )
        # LLMを呼び出して画像プロンプトを生成
        response = llm.invoke([HumanMessage(content=enhanced_template)])
        generated_prompt = response.content.strip()
        print(f"✨ サブタイトル重視プロンプト生成完了: {generated_prompt[:100]}...")
        # 生成されたプロンプトが空の場合はデフォルト値を返す
        return (
            generated_prompt
            or f"A beautiful anime-style scene of {prefecture}, Japan, depicting the theme '{sub_title}' with regional characteristics."
        )
    except Exception as e:
        print(f"⚠️ プロンプト生成エラー: {e}")
        print(
            f"🔍 Prefecture: '{prefecture}', サブタイトル: '{sub_title}', インデックス: {image_index}"
        )
        # エラー発生時のデフォルト値を返す
        return f"A beautiful anime-style scene of {prefecture or 'Japan'}, Japan, depicting the theme '{sub_title}' with regional characteristics."


def _generate_image(
    image_model: ImageGenerationModel,
    prompt: str,
    image_index: int = 0,
    total_images: int = 1,
) -> Optional[bytes]:
    """画像を生成する"""
    try:
        # ネガティブプロンプト（画像に含めたくない要素）
        negative_prompt = "text, words, letters, signs, logos, ugly, low quality, blurry, deformed, nsfw, people, characters, faces, humans, generic landscape, repetitive elements"

        print(
            f"🖼️ サブタイトル重視画像生成中 [{image_index}/{total_images}] (プロンプト: {prompt[:80]}...)"
        )

        # 画像生成モデルを呼び出して画像を生成
        response = image_model.generate_images(
            prompt=prompt,
            number_of_images=1,
            aspect_ratio="4:3",
            negative_prompt=negative_prompt,
            guidance_scale=8.0,
            seed=None,
        )

        # 生成された画像バイトを取得
        if response.images and hasattr(response.images[0], "_image_bytes"):
            print(f"✅ サブタイトル重視画像 [{image_index}/{total_images}] 生成成功")
            return response.images[0]._image_bytes
        else:
            print(f"⚠️ 画像 [{image_index}/{total_images}] の取得に失敗")
            return None

    except Exception as e:
        print(f"❌ 画像 [{image_index}/{total_images}] 生成APIエラー: {e}")
        # エラーのトレースバックを出力
        traceback.print_exc()
        return None


def generate_prefecture_image_and_get_path(
    prefecture: str,
    main_title: str,
    sub_titles: List[str],
    gcp_project_id: str,
    gcp_location: str,
    llm_model_name: str = "gemini-1.5-pro-001",
    image_gen_model_name: str = "imagen-3.0-fast-generate-001",
) -> List[str]:
    """
    都道府県とサブタイトルに基づいて、各サブタイトルの内容を反映した地域特色画像を生成します。

    Args:
        prefecture: 都道府県名
        main_title: メインタイトル
        sub_titles: サブタイトルのリスト（各画像のテーマ）
        gcp_project_id: GCPプロジェクトID
        gcp_location: GCPロケーション
        llm_model_name: 使用するLLMモデル名
        image_gen_model_name: 使用する画像生成モデル名

    Returns:
        生成された画像ファイルのパスのリスト
    """

    # 入力値の検証
    if not all([prefecture, main_title, sub_titles, gcp_project_id, gcp_location]):
        print("❌ エラー: 必須引数が不足しています。")
        return []

    # 変数の初期化
    generated_image_paths = []
    # 一時ディレクトリを作成
    temp_dir = tempfile.mkdtemp(prefix=f"subtitle_img_{prefecture}_")
    total_images = len(sub_titles)

    print(f"\n🚀 サブタイトル重視画像生成プロセス開始")
    print(f"📁 一時ディレクトリ: {temp_dir}")
    print(f"🎯 生成対象: {prefecture} - {total_images}個のサブタイトル画像")

    # モデルの初期化
    image_model, llm = _initialize_vertex_ai(
        gcp_project_id, gcp_location, image_gen_model_name, llm_model_name
    )

    if not image_model or not llm:
        print("❌ モデル初期化失敗。処理を中断します。")
        return []

    # 地域特性を一度だけ生成
    regional_characteristics = _generate_regional_characteristics(llm, prefecture)

    # 各サブタイトルに対して画像を生成
    for i, sub_title in enumerate(sub_titles):
        image_number = i + 1
        print(
            f"\n--- 🎨 サブタイトル画像 {image_number}/{total_images} 処理開始: 「{sub_title}」 ---"
        )

        # サブタイトル中心のプロンプトを生成
        image_prompt = _generate_image_prompt(
            llm,
            prefecture,
            main_title,
            sub_title,
            regional_characteristics,
            image_number,
            total_images,
        )

        # 画像を生成
        image_bytes = _generate_image(
            image_model, image_prompt, image_number, total_images
        )

        # 画像の保存
        if image_bytes:
            try:
                # ファイル名を安全な形式にする
                safe_subtitle = "".join(
                    c if c.isalnum() or c in "-_" else "_" for c in sub_title
                )[
                    :50
                ]  # サブタイトルを50文字に制限
                file_name = f"subtitle_{image_number:02d}_{safe_subtitle}.png"
                image_file_path = os.path.join(temp_dir, file_name)

                # ファイルに書き込み
                with open(image_file_path, "wb") as f:
                    f.write(image_bytes)

                generated_image_paths.append(image_file_path)
                print(f"💾 サブタイトル画像 {image_number} を保存: {image_file_path}")

            except Exception as e:
                print(f"❌ 画像 {image_number} の保存エラー: {e}")
                traceback.print_exc()
        else:
            print(
                f"⚠️ 画像 {image_number} ('{sub_title}') の生成に失敗。スキップします。"
            )

    # 完了報告
    success_count = len(generated_image_paths)
    print(f"\n🎉 サブタイトル重視画像生成処理完了")
    print(f"📊 結果: {success_count}/{total_images} 個のサブタイトル画像を正常に生成")

    if success_count < total_images:
        print(f"⚠️ {total_images - success_count} 個の画像生成に失敗しました")

    return generated_image_paths


def generate_prefecture_image_and_get_path_with_progress(
    prefecture: str,
    main_title: str,
    sub_titles: List[str],
    gcp_project_id: str,
    gcp_location: str,
    llm_model_name: str = "gemini-1.5-pro-001",
    image_gen_model_name: str = "imagen-3.0-fast-generate-001",
):
    """
    進捗状況をyieldしながらサブタイトル重視画像を生成するジェネレータ関数

    Yields:
        dict: 進捗情報 ("current", "total", "subtitle", "status", "paths")
    """

    # 入力値の検証と初期化
    if not all([prefecture, main_title, sub_titles, gcp_project_id, gcp_location]):
        print("❌ エラー: 必須引数が不足しています。")
        return

    generated_image_paths = []
    # 一時ディレクトリを作成
    temp_dir = tempfile.mkdtemp(prefix=f"subtitle_img_{prefecture}_")
    total_images = len(sub_titles)

    print(f"\n🚀 サブタイトル重視画像生成プロセス開始")
    print(f"📁 一時ディレクトリ: {temp_dir}")
    print(f"🎯 生成対象: {prefecture} - {total_images}個のサブタイトル画像")

    # モデルの初期化
    image_model, llm = _initialize_vertex_ai(
        gcp_project_id, gcp_location, image_gen_model_name, llm_model_name
    )

    if not image_model or not llm:
        print("❌ モデル初期化失敗。処理を中断します。")
        return

    # 地域特性を生成
    regional_characteristics = _generate_regional_characteristics(llm, prefecture)

    # 各サブタイトルに対して画像を生成
    for i, sub_title in enumerate(sub_titles):
        image_number = i + 1
        print(
            f"\n--- 🎨 サブタイトル画像 {image_number}/{total_images} 処理開始: 「{sub_title}」 ---"
        )

        # 進捗状況をyield
        yield {
            "current": image_number,
            "total": total_images,
            "subtitle": sub_title,
            "status": "generating_subtitle_image",
        }

        # サブタイトル中心のプロンプトを生成
        image_prompt = _generate_image_prompt(
            llm,
            prefecture,
            main_title,
            sub_title,
            regional_characteristics,
            image_number,
            total_images,
        )

        # 画像を生成
        image_bytes = _generate_image(
            image_model, image_prompt, image_number, total_images
        )

        # 画像の保存ロジック
        if image_bytes:
            try:
                safe_subtitle = "".join(
                    c if c.isalnum() or c in "-_" else "_" for c in sub_title
                )[
                    :50
                ]  # サブタイトルを50文字に制限
                file_name = f"subtitle_{image_number:02d}_{safe_subtitle}.png"
                image_file_path = os.path.join(temp_dir, file_name)

                with open(image_file_path, "wb") as f:
                    f.write(image_bytes)

                generated_image_paths.append(image_file_path)
                print(f"💾 サブタイトル画像 {image_number} を保存: {image_file_path}")

            except Exception as e:
                print(f"❌ 画像 {image_number} の保存エラー: {e}")
                traceback.print_exc()
        else:
            print(
                f"⚠️ 画像 {image_number} ('{sub_title}') の生成に失敗。スキップします。"
            )

    # 最終結果をyield
    success_count = len(generated_image_paths)
    print(f"\n🎉 サブタイトル重視画像生成処理完了")
    print(f"📊 結果: {success_count}/{total_images} 個のサブタイトル画像を正常に生成")

    yield {
        "current": total_images,
        "total": total_images,
        "subtitle": "サブタイトル画像生成完了",
        "status": "completed_subtitle_images",
        "paths": generated_image_paths,
    }
