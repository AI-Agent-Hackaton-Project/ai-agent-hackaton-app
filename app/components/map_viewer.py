import pydeck as pdk
import geopandas as gpd
import pandas as pd
import json
from utils.map_utils import limit_view_state


def create_pydeck_map(gdf, selected_region_name_on_map, current_view_state):
    """PyDeck 地図生成 (マーカーをScatterplotLayerに変更)"""
    if gdf.empty:
        return None

    geojson_features = []
    for idx, row in gdf.iterrows():
        try:

            if not row["geometry"].is_valid:
                continue

            feature = {
                "type": "Feature",
                "properties": {
                    "nam_ja": str(row.get("nam_ja", "情報なし")),
                    "nam": str(row.get("nam", "Unknown")),
                    "index": int(idx),
                    "selected": (
                        1
                        if str(row.get("nam_ja", "")) == selected_region_name_on_map
                        else 0
                    ),
                },
                "geometry": json.loads(gpd.GeoSeries([row["geometry"]]).to_json())[
                    "features"
                ][0]["geometry"],
            }
            geojson_features.append(feature)
        except Exception:

            continue
    geojson_data = {"type": "FeatureCollection", "features": geojson_features}

    geojson_layer = pdk.Layer(
        "GeoJsonLayer",
        data=geojson_data,
        id="japan-prefectures",
        pickable=True,
        stroked=True,
        filled=True,
        extruded=False,
        get_fill_color="[properties.selected ? 255 : 200, properties.selected ? 140 : 200, properties.selected ? 0 : 200, properties.selected ? 180 : 120]",
        get_line_color=[80, 80, 80, 200],
        line_width_min_pixels=1,
        auto_highlight=True,
        highlight_color=[255, 255, 0, 150],
    )

    tooltip_html = """
    <div style="background: linear-gradient(135deg, rgba(0,0,0,0.9), rgba(40,40,40,0.9)); color: white; padding: 15px; border-radius: 10px; font-family: 'Segoe UI', Arial, sans-serif; box-shadow: 0 4px 15px rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.1); min-width: 200px;">
        <div style="font-size: 18px; font-weight: bold; margin-bottom: 8px; color: #FFD700; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">🏛️ {nam_ja}</div>
        <div style="font-size: 13px; color: #B0B0B0;">📍 {nam}</div>
    </div>"""
    tooltip = {
        "html": tooltip_html,
        "style": {
            "backgroundColor": "transparent",
            "border": "none",
            "fontSize": "14px",
        },
    }

    scatter_data = []
    if selected_region_name_on_map:
        selected_gdf_rows = gdf[gdf["nam_ja"] == selected_region_name_on_map]
        if not selected_gdf_rows.empty:
            row = selected_gdf_rows.iloc[0]
            if pd.notna(row["center_x"]) and pd.notna(row["center_y"]):
                scatter_data = [
                    {
                        "position": [row["center_x"], row["center_y"]],
                        "nam_ja": row["nam_ja"],
                        "size": 5,
                    }
                ]

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=scatter_data,
        id="selected-scatter-marker",
        get_position="position",
        get_fill_color=[255, 0, 0, 220],
        get_radius="size",
        radius_units="pixels",
        pickable=False,
        stroked=True,
        get_line_color=[255, 255, 255, 180],
        line_width_min_pixels=1,
    )

    limited_view_state = limit_view_state(current_view_state)
    deck = pdk.Deck(
        layers=[geojson_layer, scatter_layer],
        initial_view_state=limited_view_state,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/light-v10",
        parameters={
            "controller": {
                "scrollZoom": False,
                "dragPan": True,
                "dragRotate": False,
                "doubleClickZoom": True,
                "touchZoom": True,
                "touchRotate": False,
                "keyboard": True,
            }
        },
    )
    return deck
