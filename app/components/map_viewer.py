import folium
import geopandas as gpd
import streamlit as st  # st.session_stateを使用するため


def create_folium_map_object(gdf: gpd.GeoDataFrame, center: list, zoom: int):
    """
    Folium地図オブジェクトを作成し、設定します。
    Args:
        gdf (gpd.GeoDataFrame): 都道府県のGeoDataFrame。
        center (list): 地図の中心座標 [緯度, 経度]。
        zoom (int): 初期ズームレベル。
    Returns:
        folium.Map: 設定済みのFolium地図オブジェクト。
    """
    folium_map = folium.Map(
        location=center,
        zoom_start=zoom,
        height=700,
        tiles="CartoDB positron",  # 背景タイル
        min_zoom=5,  # 最小ズームレベル
    )
    if gdf.empty:
        return folium_map

    # カスタムCSSを地図に追加 (フォーカスリングの非表示、attributionの非表示など)
    custom_css = """
    <style>
        path.leaflet-interactive:focus {
            outline: none !important;
            box-shadow: none !important;
        }
        path.leaflet-interactive:active {
            outline: none !important;
            box-shadow: none !important;
        }
        div.leaflet-control-attribution {
            display:none !important; 
        }
    </style>
    """
    folium_map.get_root().html.add_child(folium.Element(custom_css))

    # 各都道府県のGeoJSONとマーカーを追加
    for _, row in gdf.iterrows():
        name_en = row["nam"]
        name_ja = row["nam_ja"]
        centroid = row["center"]
        tooltip_text = f"{name_ja} ({name_en})"

        # 都道府県の中心にマーカーを追加
        folium.Marker(
            location=[centroid.y, centroid.x],
            tooltip=tooltip_text,
            icon=folium.Icon(color="blue", icon="map-marker", prefix="fa"),
        ).add_to(folium_map)

        # 都道府県のポリゴンを追加
        feature = {
            "type": "Feature",
            "geometry": row["geometry"].__geo_interface__,
            "properties": {
                "nam_ja": row["nam_ja"],
                "prefecture_tooltip_text": tooltip_text,
            },
        }
        folium.GeoJson(
            data=feature,
            name=name_ja,
            # スタイル関数: 選択された都道府県の色を変更
            style_function=lambda x: {
                "fillColor": (
                    "#4CAF50"  # 選択時の塗りつぶし色
                    if x["properties"]["nam_ja"]
                    == st.session_state.selected_prefecture_info
                    else "#A6CEE3"  # デフォルトの塗りつぶし色
                ),
                "color": (
                    "#2E7D32"  # 選択時の境界線色
                    if x["properties"]["nam_ja"]
                    == st.session_state.selected_prefecture_info
                    else "#1F78B4"  # デフォルトの境界線色
                ),
                "weight": (
                    2.5  # 選択時の境界線幅
                    if x["properties"]["nam_ja"]
                    == st.session_state.selected_prefecture_info
                    else 1.5  # デフォルトの境界線幅
                ),
                "fillOpacity": (
                    0.7  # 選択時の塗りつぶし透明度
                    if x["properties"]["nam_ja"]
                    == st.session_state.selected_prefecture_info
                    else 0.5  # デフォルトの塗りつぶし透明度
                ),
            },
            # ハイライト関数: ホバー時のスタイル
            highlight_function=lambda x: {
                "fillColor": "#4CAF50",
                "color": "#2E7D32",
                "weight": 2.5,
                "fillOpacity": 0.4,
            },
            # ツールチップ設定
            tooltip=folium.GeoJsonTooltip(
                fields=["prefecture_tooltip_text"], labels=False
            ),
            zoom_on_click=True,  # クリックでズーム
        ).add_to(folium_map)
    return folium_map


def process_map_interactions(map_data: dict, gdf: gpd.GeoDataFrame):
    """
    st_foliumからのインタラクションを処理し、セッションステートを更新します。
    Args:
        map_data (dict): st_foliumから返されるインタラクションデータ。
        gdf (gpd.GeoDataFrame): 都道府県のGeoDataFrame。
    """
    # デバッグ用に地図インタラクションデータをセッションステートに保存
    st.session_state["DEBUG_map_interaction_data"] = map_data

    if gdf.empty or not map_data:
        return

    interaction_occurred = False  # インタラクションが発生したかどうかのフラグ

    # 1. GeoJSON（ポリゴン）のクリックを処理
    clicked_geojson = map_data.get("last_active_drawing")
    if clicked_geojson and clicked_geojson.get("properties"):
        name_ja = clicked_geojson["properties"].get("nam_ja")
        if name_ja:
            row_gdf = gdf[gdf["nam_ja"] == name_ja]
            if not row_gdf.empty:
                center = row_gdf.iloc[0]["center"]
                new_center = [center.y, center.x]
                new_zoom = 8
                # 中心または選択都道府県が変更された場合のみ更新
                if (
                    st.session_state.map_center != new_center
                    or st.session_state.selected_prefecture_info != name_ja
                ):
                    st.session_state.map_center = new_center
                    st.session_state.map_zoom = new_zoom
                    st.session_state.selected_prefecture_info = name_ja
                    st.session_state.last_map_interaction_type = "click_geojson"
                    interaction_occurred = True

    # 2. マーカーのクリックを処理（GeoJSONクリックがなかった場合のみ）

    if not interaction_occurred:
        clicked_tooltip = map_data.get("last_object_clicked_tooltip")
        if clicked_tooltip:
            try:
                # ツールチップテキストから都道府県名（日本語）を抽出
                name_ja = clicked_tooltip.split(" (")[0]
                row_gdf = gdf[gdf["nam_ja"] == name_ja]
                if not row_gdf.empty:
                    center = row_gdf.iloc[0]["center"]
                    new_center = [center.y, center.x]
                    new_zoom = 8
                    # 中心、ズーム、または選択都道府県が変更された場合のみ更新
                    if (
                        st.session_state.map_center != new_center
                        or st.session_state.map_zoom != new_zoom
                        or st.session_state.selected_prefecture_info != name_ja
                    ):
                        st.session_state.map_center = new_center
                        st.session_state.map_zoom = new_zoom
                        st.session_state.selected_prefecture_info = name_ja
                        st.session_state.last_map_interaction_type = "click_marker"
                        interaction_occurred = True
            except:
                # ツールチップの解析に失敗した場合
                pass

    # インタラクションが発生した場合のみ、Streamlitアプリを再実行してUIを更新
    if interaction_occurred:
        st.rerun()
