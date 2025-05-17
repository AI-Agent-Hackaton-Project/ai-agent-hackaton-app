import streamlit as st
from streamlit_geolocation import streamlit_geolocation


def render_sidebar_controls():
    """
    サイドバーのコントロールをレンダリングし、現在のジオロケーションデータを返します。
    Returns:
        dict or None: 現在のジオロケーションデータ。
    """
    with st.sidebar:
        st.markdown("## 🗺️ コントロールと情報")
        st.markdown("#### 📍 現在位置へ移動")
        st.markdown(
            "下のアイコンをクリックすると、現在地の都道府県に地図が移動します。"
        )

        # 現在のジオロケーションを取得
        current_location = streamlit_geolocation()
        if current_location:
            st.session_state.current_geolocation_raw = current_location

        # 日本全体表示ボタン
        if st.button("🗾 日本全体表示に戻す"):
            st.session_state.map_center = [36.2048, 138.2529]
            st.session_state.map_zoom = 5
            st.session_state.selected_prefecture_info = None
            st.session_state.last_map_interaction_type = "reset_view"
            st.rerun()  # 状態変更を反映するため再実行

        st.markdown("---")
        st.markdown("#### 選択中の地域:")
        if st.session_state.selected_prefecture_info:
            st.success(f"**{st.session_state.selected_prefecture_info}**")
            st.markdown(
                f"今後、AIを活用して **{st.session_state.selected_prefecture_info}** の文化、歴史、観光地情報などを提供予定です。"
            )
        else:
            st.info("（未選択）")

        # デバッグ情報表示
        with st.expander("🛠️ デバッグ情報"):
            st.json(
                {
                    "地図の中心": st.session_state.map_center,
                    "地図のズーム": st.session_state.map_zoom,
                    "選択された都道府県": st.session_state.selected_prefecture_info,
                    "最後の操作タイプ": st.session_state.last_map_interaction_type,
                    "現在のジオロケーション(生データ)": (
                        current_location if current_location else "N/A"
                    ),
                    "map_interaction_data (st_foliumから)": st.session_state.get(
                        "DEBUG_map_interaction_data", "N/A"
                    ),
                }
            )
    return current_location
