import pydeck as pdk
from config.constants import JAPAN_BOUNDS


def limit_view_state(view_state):
    limited_longitude = max(
        JAPAN_BOUNDS["min_lon"], min(JAPAN_BOUNDS["max_lon"], view_state.longitude)
    )
    limited_latitude = max(
        JAPAN_BOUNDS["min_lat"], min(JAPAN_BOUNDS["max_lat"], view_state.latitude)
    )
    limited_zoom = max(
        JAPAN_BOUNDS["min_zoom"], min(JAPAN_BOUNDS["max_zoom"], view_state.zoom)
    )
    return pdk.ViewState(
        longitude=limited_longitude,
        latitude=limited_latitude,
        zoom=limited_zoom,
        min_zoom=JAPAN_BOUNDS["min_zoom"],
        pitch=view_state.pitch,
        bearing=view_state.bearing,
    )
