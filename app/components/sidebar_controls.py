import streamlit as st
from streamlit_geolocation import streamlit_geolocation
import time
import pydeck as pdk
from shapely.geometry import Point

from config.constants import (
    JAPAN_PREFECTURES,
    PLACEHOLDER_SELECTBOX,
    DEFAULT_SELECTED_REGION_ON_MAP,
    INITIAL_CENTER_LON,
    INITIAL_CENTER_LAT,
    INITIAL_ZOOM,
)


def render_sidebar(gdf):
    """サイドバーのレンダリングと操作処理"""
    user_location = None
    with st.sidebar:
        st.header("🗺️ コントロールと情報")
        st.markdown("#### 📍 現在位置へ移動")
        st.markdown(
            "下のアイコンをクリックすると、現在地の都道府県に地図が移動します。"
        )

        user_location = streamlit_geolocation()

        actual_prefectures = JAPAN_PREFECTURES
        if (
            not actual_prefectures
            and gdf is not None
            and not gdf.empty
            and "nam_ja" in gdf.columns
        ):
            actual_prefectures = sorted(gdf["nam_ja"].unique().tolist())

        selectbox_options = [PLACEHOLDER_SELECTBOX] + (
            actual_prefectures if actual_prefectures else []
        )
        current_selectbox_display_value = st.session_state.get(
            "selectbox_value", PLACEHOLDER_SELECTBOX
        )

        try:
            idx = selectbox_options.index(current_selectbox_display_value)
        except ValueError:
            idx = 0
            st.session_state.selectbox_value = PLACEHOLDER_SELECTBOX

        sel_via_selectbox = st.selectbox(
            "地域選択",
            options=selectbox_options,
            index=idx,
            key="sb_region_selection_sidebar",  # keyは必須
            help="ドロップダウンから地域を選択します。",
        )

        if st.button(
            "🗾 地図リセット",
            key="reset_map_button_sidebar",
            help="選択を解除し、地図を初期状態に戻します。",
            use_container_width=True,
        ):
            st.session_state.selected_region_on_map = DEFAULT_SELECTED_REGION_ON_MAP
            st.session_state.selected_prefecture_info = None
            st.session_state.map_view_state = pdk.ViewState(
                longitude=INITIAL_CENTER_LON,
                latitude=INITIAL_CENTER_LAT,
                zoom=INITIAL_ZOOM,
                pitch=0,
                bearing=0,
            )
            st.session_state.last_clicked_time = 0.0
            st.session_state.selectbox_value = PLACEHOLDER_SELECTBOX
            st.session_state.last_map_interaction_type = "reset_button"
            st.rerun()

        if sel_via_selectbox != st.session_state.selectbox_value:
            st.session_state.selectbox_value = sel_via_selectbox
            if sel_via_selectbox != PLACEHOLDER_SELECTBOX:
                st.session_state.selected_region_on_map = sel_via_selectbox
                st.session_state.selected_prefecture_info = sel_via_selectbox
                st.session_state.last_map_interaction_type = "selectbox_selection"

                if gdf is not None and not gdf.empty:
                    matched_row = gdf[gdf["nam_ja"] == sel_via_selectbox]
                    if not matched_row.empty:
                        pref_data = matched_row.iloc[0]
                        if "center" in pref_data and isinstance(
                            pref_data["center"], Point
                        ):
                            pref_center_pt = pref_data["center"]
                            st.session_state.map_view_state = pdk.ViewState(
                                latitude=pref_center_pt.y,
                                longitude=pref_center_pt.x,
                                zoom=max(st.session_state.map_view_state.zoom, 6),
                                pitch=st.session_state.map_view_state.pitch,
                                bearing=st.session_state.map_view_state.bearing,
                                transition_duration=500,
                                transition_interruption="allowed",
                            )
            else:
                st.session_state.selected_region_on_map = DEFAULT_SELECTED_REGION_ON_MAP
                st.session_state.selected_prefecture_info = None
                st.session_state.map_view_state = pdk.ViewState(
                    longitude=INITIAL_CENTER_LON,
                    latitude=INITIAL_CENTER_LAT,
                    zoom=INITIAL_ZOOM,
                    pitch=0,
                    bearing=0,
                )
            st.session_state.last_clicked_time = time.time()
            st.rerun()

        st.markdown("---")
        if st.session_state.get("selected_prefecture_info"):
            st.markdown("#### 選択中の地域:")
            st.success(f"🎯 **{st.session_state.selected_prefecture_info}**")
        else:
            st.info(
                "🖱️ 地図上の都道府県をクリックするか、上のメニューから選択してください。"
            )
    return user_location
