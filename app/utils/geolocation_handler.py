import streamlit as st
import geopandas as gpd
from shapely.geometry import Point


def process_geolocation_data(current_location: dict, gdf: gpd.GeoDataFrame):
    """
    ジオロケーションデータを処理し、必要に応じてセッションステートを更新します。
    Args:
        current_location (dict): streamlit_geolocationから返される現在の位置情報。
        gdf (gpd.GeoDataFrame): 都道府県のGeoDataFrame。
    """
    # データが空の場合、またはジオロケーションが利用できない場合は処理しない
    if (
        gdf.empty
        or not current_location
        or not (current_location.get("latitude") and current_location.get("longitude"))
    ):
        return

    # ジオロケーションデータが前回と異なる場合のみ処理
    if current_location != st.session_state.last_location_data:
        st.session_state.last_location_data = current_location
        user_point = Point(current_location["longitude"], current_location["latitude"])

        # ジオメトリカラムが存在しない、または全てがNoneの場合は処理しない
        if "geometry" not in gdf.columns or gdf["geometry"].isnull().all():
            return

        # 現在地がどの都道府県に含まれるか検索
        matched_row = gdf[gdf.geometry.contains(user_point)]

        if not matched_row.empty:
            prefecture_data = matched_row.iloc[0]
            pref_center = prefecture_data["center"]
            new_center = [pref_center.y, pref_center.x]
            new_zoom = 8  # 都道府県の中心にズームイン

            # 地図の中心、ズーム、または選択都道府県が変更された場合のみ更新
            if (
                st.session_state.map_center != new_center
                or st.session_state.map_zoom != new_zoom
                or st.session_state.selected_prefecture_info
                != prefecture_data["nam_ja"]
            ):
                st.session_state.map_center = new_center
                st.session_state.map_zoom = new_zoom
                st.session_state.selected_prefecture_info = prefecture_data["nam_ja"]
                st.session_state.last_map_interaction_type = "geolocation_update"
                st.rerun()  # 状態変更を反映するため再実行
        elif st.session_state.last_map_interaction_type != "geolocation_outside_japan":
            # 日本の都道府県範囲外の場合
            with st.sidebar:
                st.warning("現在地が日本の都道府県の範囲外のようです。")
            st.session_state.last_map_interaction_type = "geolocation_outside_japan"
