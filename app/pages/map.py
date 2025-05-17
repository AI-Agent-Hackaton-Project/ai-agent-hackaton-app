# pages/map_view.py
import streamlit as st
from streamlit_folium import st_folium
from components.map_viewer import create_folium_map_object, process_map_interactions
from components.sidebar_controls import render_sidebar_controls
from utils.data_loader import load_and_prepare_geojson
from utils.state_manager import initialize_session_state
from utils.geolocation_handler import process_geolocation_data


st.set_page_config(layout="wide")


def map_page():
    st.title("🗾 日本の都道府県マップ")

    # GeoJSONデータを読み込み、準備
    gdf = load_and_prepare_geojson()
    # セッションステートを初期化
    initialize_session_state()

    # サイドバーコントロールをレンダリングし、ジオロケーションデータを取得
    current_geo_location = render_sidebar_controls()
    # ジオロケーションデータに基づいて地図を更新
    process_geolocation_data(current_geo_location, gdf)

    # Folium地図オブジェクトを作成
    folium_map_instance = create_folium_map_object(
        gdf, st.session_state.map_center, st.session_state.map_zoom
    )

    # StreamlitにFolium地図を表示し、ユーザーインタラクションデータを取得
    map_interaction_data = st_folium(
        folium_map_instance,
        width="100%",
        height="100%",
        center=st.session_state.map_center,  # 地図の中心をセッションステートから設定
        # 取得したいインタラクションデータを指定
        returned_objects=[
            "last_object_clicked_tooltip",
            "last_active_drawing",
            "center",
        ],
    )

    # 地図インタラクションデータを処理
    process_map_interactions(map_interaction_data, gdf)


if __name__ == "__main__":
    map_page()
