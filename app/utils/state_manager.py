import streamlit as st
import pydeck as pdk
import time
from shapely.geometry import Point
import geopandas as gpd

from config.constants import (
    DEFAULT_SELECTED_REGION_ON_MAP,
    INITIAL_CENTER_LON,
    INITIAL_CENTER_LAT,
    INITIAL_ZOOM,
    PLACEHOLDER_SELECTBOX,
)


def initialize_session_state():
    """セッション状態変数を初期化"""
    defaults = {
        "selected_region_on_map": DEFAULT_SELECTED_REGION_ON_MAP,
        "selected_prefecture_info": None,
        "map_view_state": pdk.ViewState(
            longitude=INITIAL_CENTER_LON,
            latitude=INITIAL_CENTER_LAT,
            zoom=INITIAL_ZOOM,
            pitch=0,
            bearing=0,
        ),
        "last_clicked_time": 0.0,
        "selectbox_value": PLACEHOLDER_SELECTBOX,
        "last_location_data": None,
        "last_map_interaction_type": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def process_selected_feature(selected_feature_data, gdf: gpd.GeoDataFrame):
    """地図上でクリックされた地物の情報を処理"""
    current_time = time.time()
    if not selected_feature_data or not isinstance(selected_feature_data, dict):
        return False, None

    feature_properties = (
        selected_feature_data.get("properties")
        if "properties" in selected_feature_data
        else selected_feature_data
    )
    if not isinstance(feature_properties, dict) or "nam_ja" not in feature_properties:
        return False, None

    region_name = feature_properties.get("nam_ja")
    if not region_name:
        return False, None

    current_selected_on_map = st.session_state.get("selected_region_on_map")
    time_since_last_click = current_time - st.session_state.get(
        "last_clicked_time", 0.0
    )
    is_new_region = region_name != current_selected_on_map

    can_process_click = (is_new_region and time_since_last_click > 0.5) or (
        not is_new_region and time_since_last_click > 0.2
    )

    if can_process_click:
        st.session_state.selected_region_on_map = region_name
        st.session_state.selected_prefecture_info = region_name
        st.session_state.selectbox_value = region_name
        st.session_state.last_clicked_time = current_time
        st.session_state.last_map_interaction_type = "map_click"

        matched_row = gdf[gdf["nam_ja"] == region_name]
        if not matched_row.empty:
            pref_data = matched_row.iloc[0]
            if "center" in pref_data and isinstance(pref_data["center"], Point):
                pref_center_pt = pref_data["center"]
                st.session_state.map_view_state = pdk.ViewState(
                    latitude=pref_center_pt.y,
                    longitude=pref_center_pt.x,
                    # zoom=st.session_state.map_view_state.zoom, # 変更前: 現在のズームレベルを維持
                    zoom=max(
                        st.session_state.map_view_state.zoom, 6
                    ),  # <<< 変更点: selectboxと同様に最小ズームレベル6に設定
                    pitch=st.session_state.map_view_state.pitch,
                    bearing=st.session_state.map_view_state.bearing,
                    transition_duration=500,
                    transition_interruption="allowed",
                )
        return True, region_name
    return False, region_name


def process_geolocation_data(current_location: dict, gdf: gpd.GeoDataFrame):
    """ユーザーの現在地情報を処理し、地図を更新"""
    if (
        gdf.empty
        or not current_location
        or not (current_location.get("latitude") and current_location.get("longitude"))
    ):
        if (
            not current_location
            and st.session_state.get("last_location_data") is not None
        ):
            st.session_state.last_location_data = None
        return

    if current_location == st.session_state.get("last_location_data"):
        return

    st.session_state.last_location_data = current_location
    user_point = Point(current_location["longitude"], current_location["latitude"])

    if "geometry" not in gdf.columns or gdf["geometry"].isnull().all():
        return

    matched_row = gdf[gdf.geometry.contains(user_point)].copy()

    if not matched_row.empty:
        prefecture_data = matched_row.iloc[0]
        current_pref_name = prefecture_data["nam_ja"]

        if "center" not in prefecture_data or not isinstance(
            prefecture_data["center"], Point
        ):
            return

        pref_center_point = prefecture_data["center"]

        state_changed = (
            st.session_state.map_view_state.latitude != pref_center_point.y
            or st.session_state.map_view_state.longitude != pref_center_point.x
            or st.session_state.selected_region_on_map != current_pref_name
        )

        if state_changed:
            st.session_state.map_view_state = pdk.ViewState(
                latitude=pref_center_point.y,
                longitude=pref_center_point.x,
                zoom=max(st.session_state.map_view_state.zoom, 7),
                pitch=st.session_state.map_view_state.pitch,
                bearing=st.session_state.map_view_state.bearing,
                transition_duration=1000,
                transition_interruption="allowed",
            )
            st.session_state.selected_region_on_map = current_pref_name
            st.session_state.selected_prefecture_info = current_pref_name
            st.session_state.selectbox_value = current_pref_name
            st.session_state.last_map_interaction_type = "geolocation_update"
            st.toast(f"現在地 ({current_pref_name}) に地図を移動しました。")
            st.rerun()
    else:
        if st.session_state.last_map_interaction_type != "geolocation_outside_japan":
            st.session_state.last_map_interaction_type = "geolocation_outside_japan"
            st.session_state.selected_region_on_map = DEFAULT_SELECTED_REGION_ON_MAP
            st.session_state.selected_prefecture_info = None
            st.session_state.selectbox_value = PLACEHOLDER_SELECTBOX
            st.rerun()
