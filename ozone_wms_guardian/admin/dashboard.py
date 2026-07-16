"""Guardian dashboard renderer for Streamlit."""

import streamlit as st


def render_guardian_dashboard(health, alerts, suggestions, recovery, tuner, ozone):
    """Render the Guardian Ops Center dashboard."""

    # Health gauge
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        health_pct = int(health * 100)
        health_color = "#00ff88" if health > 0.8 else "#ffd93d" if health > 0.5 else "#ff6b6b"
        st.markdown(f"""
        <div style="text-align:center; padding:20px; background:rgba(10,20,40,0.6); border-radius:16px; border:1px solid {health_color}40;">
            <div style="font-size:0.9rem; color:#8892b0; margin-bottom:8px;">SYSTEM HEALTH</div>
            <div style="font-size:3rem; font-weight:bold; color:{health_color}; text-shadow:0 0 20px {health_color}40;">{health_pct}%</div>
            <div style="font-size:0.85rem; color:{health_color}; margin-top:4px;">{ozone.get('status', 'UNKNOWN')}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Metrics
    if tuner:
        m_cols = st.columns(4)
        m_cols[0].metric("Total SKUs", tuner.get("total_skus", 0))
        m_cols[1].metric("Total Stock", tuner.get("total_stock", 0))
        m_cols[2].metric("Pending Orders", tuner.get("pending_orders", 0))
        m_cols[3].metric("Queue Items", tuner.get("queued_items", 0))

    # Alerts
    st.subheader("🚨 Active Alerts")
    if alerts:
        for alert in alerts:
            level_color = {"CRITICAL": "#ff6b6b", "WARNING": "#ffd93d", "INFO": "#64ffda"}.get(alert.get("level", "INFO"), "#64ffda")
            st.markdown(f"""
            <div style="border-left:3px solid {level_color}; padding:10px 15px; margin:8px 0; background:rgba(255,255,255,0.03); border-radius:0 8px 8px 0;">
                <span style="color:{level_color}; font-weight:bold; font-size:0.85rem;">{alert.get('level', 'INFO')}</span>
                <span style="color:#ccd6f6; margin-left:10px;">{alert.get('message', '')}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No active alerts — all systems nominal")

    # Suggestions
    if suggestions:
        st.subheader("💡 Suggestions")
        for suggestion in suggestions:
            st.markdown(f"<div style='color:#8892b0; font-size:0.9rem; margin:4px 0;'>• {suggestion}</div>", unsafe_allow_html=True)

    # Recovery actions
    if recovery:
        st.subheader("🔧 Recovery Actions")
        for action in recovery:
            st.markdown(f"<div style='color:#ff6b6b; font-size:0.9rem; margin:4px 0;'>⚠️ {action}</div>", unsafe_allow_html=True)

    # Ozone status
    st.divider()
    st.subheader("🌐 Ozone Status")
    ozone_cols = st.columns(3)
    ozone_cols[0].metric("Uptime", ozone.get("uptime", "N/A"))
    ozone_cols[1].metric("Last Check", ozone.get("last_check", "N/A"))
    ozone_cols[2].metric("Next Check", ozone.get("next_check", "N/A"))
