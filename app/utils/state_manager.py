import streamlit as st


def initialize_session_state():
    """
    セッションステート変数を初期化します。
    """
    defaults = {
        "map_center": [36.2048, 138.2529],  # 日本の初期中心座標
        "map_zoom": 5,  # 初期ズームレベル
        "selected_prefecture_info": None,  # 選択された都道府県情報
        "last_map_interaction_type": "initial_load",  # 最後の地図操作タイプ
        "last_location_data": None,  # 最後のジオロケーションデータ
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
