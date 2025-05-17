import geopandas as gpd
import streamlit as st


@st.cache_data
def load_and_prepare_geojson():
    """
    GeoJSONを読み込み、ジオメトリを簡素化し、重心を計算します。
    Returns:
        gpd.GeoDataFrame: 準備されたGeoDataFrame。
    """

    # api: "https://raw.githubusercontent.com/dataofjapan/land/master/japan.geojson"
    gdf = gpd.read_file("app/data/japan.geojson")
    original_crs = gdf.crs

    # # パフォーマンス向上のため、ジオメトリを積極的に簡素化 (許容誤差を増やす)
    gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.01)

    # 無効または空のジオメトリを持つ行を除外
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.is_valid].copy()
    # CRS情報が失われた場合、元のCRSを再設定
    if gdf.crs is None and original_crs is not None:
        gdf = gdf.set_crs(original_crs, allow_override=True)

    # 各都道府県の重心を計算
    gdf["center"] = gdf["geometry"].centroid
    return gdf
