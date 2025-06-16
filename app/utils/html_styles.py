# 記事用スタイル定義
ARTICLE_STYLES = {
    "body": """
        font-family: 'Noto Sans JP', 'Hiragino Kaku Gothic Pro', 'ヒラギノ角ゴ Pro W3', Meiryo, sans-serif; 
        margin: 0; 
        padding: 0; 
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        color: #2c3e50; 
        line-height: 1.8;
    """,
    "container": """
        max-width: 900px; 
        margin: 0 auto; 
        background-color: #ffffff;
        border-radius: 12px;
        overflow: hidden;
        margin-top: 30px;
        margin-bottom: 30px;
    """,
    "header": """
        padding:  60px 40px 0;
        text-align: center;
        color: white;
    """,
    "main_title": """
        font-family: 'Playfair Display', 'Times New Roman', serif; 
        font-size: 2em; 
        font-weight: 700; 
        margin: 0 0 20px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        letter-spacing: -1px;
    """,
    "aphorism": """
        font-size: 1.3em; 
        font-style: italic; 
        opacity: 0.9;
        padding: 20px;
        color: #34495e;
        background: rgba(255,255,255,0.1);
        border-radius: 0 0 8px 8px;
        border: 2px solid #6f92a9;
    """,
    "aphorism_title": """
        text-align: left;
        font-size: 1.3rem;
        margin-top: 20px;
        font-weight: 600;
        background: #46607a;
        padding: 5px 20px;
        border-radius: 8px 8px 0 0;
    """,
    "content": """
        padding: 50px;
        line-height: 2.0;
    """,
    "section": """
        margin-bottom: 60px;
        padding: 40px;
        background: #fafbfc;
        border-radius: 8px;
        border-left: 5px solid #6f92a9;
        line-height: 2.0;
    """,
    "subtitle": """
        font-size: 1.8em; 
        font-weight: 600; 
        color: #2c3e50;
        margin: 0 0 25px 0;
        padding-bottom: 15px;
        border-bottom: 2px solid #e9ecef;
    """,
    "text": """
        font-size: 1.1em; 
        line-height: 2.0; 
        color: #34495e;
        text-align: justify;
        margin-bottom: 25px;
        margin-top: 10px;
        padding: 10px 0;
    """,
    "image": """
        width: 100%; 
        max-width: 100%; 
        height: auto; 
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 30px 0;
        transition: transform 0.3s ease;
    """,
    "main_image": """
        width: 100%; 
        max-width: 100%; 
        object-fit: cover; 
        border-radius: 0;
        margin: 0;
    """,
    "error": """
        background: #ffe6e6;
        padding: 20px;
        margin: 20px;
        border-radius: 8px;
    """,
    "strong": """
        color: #2c3e50;
        font-weight: 600;
        text-decoration: underline;
        text-decoration-color: #3498db;
        text-decoration-thickness: 2px;
        text-underline-offset: 3px;
    """,
}
