import streamlit as st
import geopandas as gpd
from shapely.geometry import Point
from config.constants import INITIAL_CENTER_LON, INITIAL_CENTER_LAT


@st.cache_data
def load_geojson():
    urls = [
        "https://raw.githubusercontent.com/dataofjapan/land/master/japan.geojson",
    ]
    gdf = None
    for url in urls:
        try:
            gdf = gpd.read_file(url)
            if not gdf.empty:
                break
        except Exception:
            pass

    if gdf is None or gdf.empty:
        st.error("すべてのGeoJSON URLからデータをロードできませんでした。")
        return gpd.GeoDataFrame()

    column_mappings = {
        "name_ja": "nam_ja",
        "NAME_JA": "nam_ja",
        "pref_name": "nam_ja",
        "name": "nam",
        "NAME": "nam",
        "pref": "nam",
        "prefecture": "nam",
    }
    for old_col, new_col in column_mappings.items():
        if old_col in gdf.columns and new_col not in gdf.columns:
            gdf[new_col] = gdf[old_col]

    if "nam_ja" not in gdf.columns:
        gdf["nam_ja"] = gdf.index.astype(str) + "番地域"
    if "nam" not in gdf.columns:
        gdf["nam"] = gdf["nam_ja"]

    gdf["nam_ja"] = gdf["nam_ja"].astype(str).replace("", "情報なし").fillna("情報なし")
    gdf["nam"] = gdf["nam"].astype(str).replace("", "Unknown").fillna("Unknown")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    elif gdf.crs.to_string().upper() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    if "geometry" not in gdf.columns or gdf["geometry"].isnull().all():
        st.error("ジオメトリ列が見つからないか、すべて無効です。")
        return gpd.GeoDataFrame()

    gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        st.error("有効なジオメトリがありません。")
        return gpd.GeoDataFrame()

    gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.01, preserve_topology=True)

    try:
        gdf["center"] = gdf["geometry"].centroid
        gdf["center_x"] = gdf["center"].x
        gdf["center_y"] = gdf["center"].y
    except Exception:
        default_center_point = Point(INITIAL_CENTER_LON, INITIAL_CENTER_LAT)
        gdf["center"] = [default_center_point for _ in range(len(gdf))]
        gdf["center_x"] = INITIAL_CENTER_LON
        gdf["center_y"] = INITIAL_CENTER_LAT
    return gdf
