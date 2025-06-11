import os
import re
import base64
from typing import List, Optional
from .html_styles import ARTICLE_STYLES


def process_markdown_text(text: str, text_style: str) -> str:
    """Markdownの**強調**をHTMLのstrongタグに変換し、改行を処理"""
    if not text:
        return ""

    # **強調**をstrongタグに変換（スタイル付き）
    strong_style = ARTICLE_STYLES.get("strong", "")
    text = re.sub(
        r"\*\*(.*?)\*\*", rf'<strong style="{strong_style}">\1</strong>', text
    )

    # 改行を処理：\n\nを段落区切りに、\nを<br>に
    paragraphs = text.split("\n\n")
    processed_paragraphs = []

    for paragraph in paragraphs:
        if paragraph.strip():
            # 段落内の単一改行を<br>に変換
            paragraph = paragraph.replace("\n", "<br>")
            # 段落間により多くの空白を追加
            processed_paragraphs.append(
                f'<p style="{text_style} margin-bottom: 30px; margin-top: 15px;">{paragraph}</p>'
            )

    return "".join(processed_paragraphs)


def encode_image(image_path: str) -> Optional[str]:
    """画像をbase64エンコード"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        (f"画像エンコードエラー: {e}")
        return None


def build_html_article(
    article_title: str,
    subtitles: List[str],
    blocks: List[str],
    main_img: Optional[str] = None,
    sub_imgs: Optional[List[str]] = None,
    aphorism: Optional[str] = None,
    error: Optional[str] = None,
) -> str:
    """記事のHTMLを構築する"""

    styles = ARTICLE_STYLES

    # HTMLコンテンツの構築
    html_parts = [
        f'<html><head><meta charset="UTF-8"><title>{article_title}</title></head>',
        f'<body style="{styles["body"]}">',
        f'<div style="{styles["container"]}">',
        f'<div style="{styles["header"]}">',
        f'<h1 style="{styles["main_title"]}">{article_title}</h1>',
    ]

    # メイン画像の追加
    if main_img and os.path.exists(main_img):
        encoded_img = encode_image(main_img)
        if encoded_img:
            html_parts.append(
                f'<img src="data:image/png;base64,{encoded_img}" style="{styles["main_image"]}" alt="メイン画像">'
            )

    # 名言の追加
    if aphorism:
        html_parts.append(f'<div style="{styles["aphorism"]}">{aphorism}</div>')

    html_parts.append("</div>")  # header終了
    html_parts.append(f'<div style="{styles["content"]}">')

    # 各セクションの追加
    sub_imgs = sub_imgs or []
    for i, (subtitle, block) in enumerate(zip(subtitles, blocks)):
        html_parts.append(f'<div style="{styles["section"]}">')
        html_parts.append(f'<h2 style="{styles["subtitle"]}">{subtitle}</h2>')

        # サブタイトル画像の追加
        if i < len(sub_imgs) and os.path.exists(sub_imgs[i]):
            encoded_img = encode_image(sub_imgs[i])
            if encoded_img:
                html_parts.append(
                    f'<img src="data:image/png;base64,{encoded_img}" style="{styles["image"]}" alt="{subtitle}の画像">'
                )

        # テキストコンテンツの追加
        processed_text = process_markdown_text(block, styles["text"])
        html_parts.append(processed_text)
        html_parts.append("</div>")

    html_parts.extend(["</div>", "</div>", "</body>", "</html>"])

    final_html = "".join(html_parts)

    # エラー情報の追加
    if error:
        final_html += (
            f'<div style="{styles["error"]}"><strong>エラー情報:</strong> {error}</div>'
        )

    return final_html
