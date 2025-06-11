import streamlit as st
from utils.state_manager import (
    initialize_session_state,
    process_selected_feature,
    process_geolocation_data,
)
from utils.data_loader import load_geojson
from components.sidebar_controls import render_sidebar
from components.map_viewer import create_pydeck_map


def map_section():

    initialize_session_state()

    gdf = load_geojson()
    if gdf.empty:
        return

    current_user_location = render_sidebar(gdf)

    if not gdf.empty:
        process_geolocation_data(current_user_location, gdf)

    map_render_selection = st.session_state.selected_region_on_map
    map_render_view_state = st.session_state.map_view_state

    deck_obj = create_pydeck_map(gdf, map_render_selection, map_render_view_state)

    event_info = None
    if deck_obj:
        event_info = st.pydeck_chart(
            deck_obj,
            use_container_width=True,
            key="jp_map_interactive_final_geo_v5",
            on_select="rerun",
            selection_mode="single-object",
        )

    clicked_feature_props = None

    if event_info and hasattr(event_info, "selection") and event_info.selection:
        payload = event_info.selection
        if (
            isinstance(payload, dict)
            and "objects" in payload
            and isinstance(payload["objects"], dict)
        ):
            features_list = payload["objects"].get("japan-prefectures")
            if isinstance(features_list, list) and len(features_list) > 0:
                raw_feature = features_list[0]
                if isinstance(raw_feature, dict) and "properties" in raw_feature:
                    clicked_feature_props = raw_feature["properties"]
                else:

                    clicked_feature_props = raw_feature

    elif event_info and isinstance(event_info, dict) and not clicked_feature_props:
        if event_info.get("layer_id") == "japan-prefectures":

            clicked_object_or_props = event_info.get("object")
            if clicked_object_or_props and isinstance(clicked_object_or_props, dict):

                clicked_feature_props = clicked_object_or_props

    if clicked_feature_props:
        success, _ = process_selected_feature(clicked_feature_props, gdf)
        if success:
            st.rerun()
