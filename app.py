import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analyzer import Analyzer
from helpers import (
    compute_trend_score, spike_alerts, auto_insights,
    predict_lstm, chatbot_reply, to_excel
)

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="Google Search Trends Analysis",
    layout="wide",
    page_icon="📈"
)

# ================================================================
# GLOBAL CSS
# ================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: #fff;
}
.card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 22px;
    border: 1px solid rgba(255,255,255,0.1);
}
.metric-box {
    background: rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.metric-box h1 { font-size: 2rem; margin: 0; }
.metric-box p  { margin: 0; color: #aaa; font-size: 0.85rem; }

section[data-testid="stSidebar"] { background: #0d1117 !important; }

.stButton>button {
    background: linear-gradient(90deg, #ff7e5f, #feb47b);
    border-radius: 10px;
    color: #fff !important;
    font-weight: 700;
    border: none;
    transition: 0.3s;
}
.stButton>button:hover { transform: scale(1.05); }

.chat-user {
    background: #2d4356;
    border-radius: 12px 12px 2px 12px;
    padding: 10px 14px;
    margin: 6px 0;
    max-width: 80%;
    float: right;
    clear: both;
}
.chat-bot {
    background: rgba(255,255,255,0.07);
    border-radius: 12px 12px 12px 2px;
    padding: 10px 14px;
    margin: 6px 0;
    max-width: 80%;
    float: left;
    clear: both;
}
.chat-wrap { overflow: auto; margin-bottom: 10px; }
h1, h2, h3 { color: #ffffff; }

/* Hide Streamlit Deploy button and top-right toolbar */
#MainMenu { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.stDeployButton { display: none !important; }
footer { visibility: hidden; }
div[data-testid="stToolbar"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ================================================================
# INIT ANALYZER & SESSION STATE
# ================================================================
@st.cache_resource
def get_analyzer():
    return Analyzer()

analyzer = get_analyzer()

for key, default in [
    ("search_history", []),
    ("chat_history",   []),
    ("last_data",      None),
    ("last_keywords",  []),
    ("region_data",    None),
    ("forecast",       None),
    ("chat_prefill",   ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ================================================================
# HEADER
# ================================================================
st.title("📈 Google Search Trends Analysis")
st.markdown("### Smart Trend Analytics · AI Forecast · Competitor Intelligence · Chatbot")

# ================================================================
# SIDEBAR
# ================================================================
st.sidebar.markdown("""
<div style="text-align:center; padding: 10px 0 16px 0;">
    <h2 style="color:#ff7e5f; margin:0;">📈 Google Search</h2>
    <h2 style="color:#feb47b; margin:0;">Trends Analysis</h2>
    <p style="color:#aaa; font-size:0.78rem; margin-top:4px;">AI-Powered Trend Intelligence</p>
    <hr style="border-color:rgba(255,255,255,0.1); margin-top:10px;">
</div>
""", unsafe_allow_html=True)

st.sidebar.header("⚙️ Controls")
kw_input = st.sidebar.text_input(
    "Keywords (comma-separated, max 5)",
    "AI, Cloud Computing, Data Science"
)
tf  = st.sidebar.selectbox("Timeframe", ["today 12-m", "today 5-y", "today 1-m", "now 7-d"])
geo = st.sidebar.text_input("Country Code (e.g. IN, US — blank = Global)", "")

st.sidebar.markdown("---")

# ---- Live Suggestions ----
st.sidebar.subheader("💡 Live Suggestions")
st.sidebar.caption("Click any suggestion to ask the chatbot instantly:")
first_kw = kw_input.split(",")[0].strip()
if first_kw:
    live_sugg, _ = analyzer.suggest(first_kw)
    for s in live_sugg[:5]:
        if st.sidebar.button(f"🔎 {s['title']}", key=f"sugg_{s['title']}"):
            st.session_state.chat_prefill = f"Tell me about trends for {s['title']}"

st.sidebar.markdown("---")

# ---- Search History ----
st.sidebar.subheader("🕓 Search History")
if st.session_state.search_history:
    for h in reversed(st.session_state.search_history[-8:]):
        st.sidebar.caption(f"• {h}")
    if st.sidebar.button("🗑️ Clear History"):
        st.session_state.search_history = []
        st.rerun()
else:
    st.sidebar.caption("No searches yet.")

# ================================================================
# GENERATE DASHBOARD
# ================================================================
if st.sidebar.button("🚀 Generate Full Dashboard"):

    keywords = [k.strip() for k in kw_input.split(",") if k.strip()][:5]
    geo_code = geo.strip().upper()

    if not keywords:
        st.error("Please enter at least one keyword.")
        st.stop()

    # Save to history
    entry = f"{', '.join(keywords)} [{tf}]"
    if entry not in st.session_state.search_history:
        st.session_state.search_history.append(entry)

    # ---------- Fetch trend data ----------
    with st.spinner("Fetching & analyzing search data..."):
        data, err = analyzer.interest(keywords, tf, geo_code)

    if err:
        st.error(f"API Error: {err}")
        st.stop()

    if data is None or data.empty:
        st.warning("No data returned. Try a different keyword or wait 30s to avoid rate limiting.")
        st.stop()

    data = data.drop(columns=['isPartial'], errors='ignore')
    st.session_state.last_data     = data
    st.session_state.last_keywords = keywords

    # ================================================================
    # SECTION 1 — KPI TILES
    # ================================================================
    st.markdown("## 📊 Keyword KPI Dashboard")
    kpi_cols = st.columns(len(keywords))
    for i, kw in enumerate(keywords):
        if kw in data.columns:
            score, label_s, delta = compute_trend_score(data[kw])
            with kpi_cols[i]:
                st.markdown(f"""<div class="metric-box">
                    <h1>{score}</h1>
                    <p>Trend Score</p>
                    <p><b>{kw}</b></p>
                    <p>{label_s} | Δ {delta:+.1f}%</p>
                </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ================================================================
    # SECTION 2 — SPIKE ALERTS
    # ================================================================
    alerts = spike_alerts(data, keywords)
    if alerts:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔔 Trend Spike Alerts")
        for a in alerts:
            st.warning(a)
        st.markdown('</div>', unsafe_allow_html=True)

    # ================================================================
    # SECTION 3 — INTEREST OVER TIME
    # ================================================================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Interest Over Time")
    fig_time = px.line(
        data, x=data.index, y=keywords,
        template="plotly_dark",
        title="Historical Search Interest",
        labels={'value': 'Search Volume (0–100)', 'date': 'Date'}
    )
    st.plotly_chart(fig_time, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ================================================================
    # SECTION 4 — GAUGE CHARTS
    # ================================================================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🎯 Popularity Gauge")
    gcols = st.columns(len(keywords))
    for i, kw in enumerate(keywords):
        if kw in data.columns:
            score, _, _ = compute_trend_score(data[kw])
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={'text': kw},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar':  {'color': "#ff7e5f"},
                    'steps': [
                        {'range': [0,  40], 'color': '#1a1a2e'},
                        {'range': [40, 70], 'color': '#16213e'},
                        {'range': [70,100], 'color': '#0f3460'},
                    ]
                }
            ))
            fig_g.update_layout(
                template="plotly_dark", height=250,
                margin=dict(t=50, b=20, l=20, r=20)
            )
            gcols[i].plotly_chart(fig_g, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ================================================================
    # SECTION 5 — PIE CHART (Market Share)
    # ================================================================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Market Share (Average Interest)")
    avgs    = {kw: round(data[kw].mean(), 1) for kw in keywords if kw in data.columns}
    fig_pie = px.pie(
        names=list(avgs.keys()), values=list(avgs.values()),
        template="plotly_dark",
        color_discrete_sequence=px.colors.sequential.Plasma_r
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ================================================================
    # SECTION 6 — PERIOD COMPARISON
    # ================================================================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🕒 First Half vs Second Half Comparison")
    mid         = len(data) // 2
    first_half  = data.iloc[:mid].mean().round(1)
    second_half = data.iloc[mid:].mean().round(1)
    valid_kws   = [k for k in keywords if k in data.columns]
    comp_df     = pd.DataFrame(
        {'First Period': first_half, 'Recent Period': second_half}
    ).loc[valid_kws]
    fig_comp = px.bar(
        comp_df.T, barmode='group', template='plotly_dark',
        title="Period Comparison (First vs Recent)",
        labels={'value': 'Avg Interest', 'index': 'Keyword'}
    )
    st.plotly_chart(fig_comp, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ================================================================
    # SECTION 7 — CORRELATION MATRIX
    # ================================================================
    if len(keywords) > 1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Keyword Correlation Matrix")
        valid = [k for k in keywords if k in data.columns]
        corr  = data[valid].corr().round(2)
        fig_corr = px.imshow(
            corr, text_auto=True,
            color_continuous_scale='RdBu_r',
            template='plotly_dark',
            title='How closely do keywords move together?'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ================================================================
    # SECTION 8 — LSTM FORECAST
    # ================================================================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🤖 Deep Learning AI Forecast (LSTM)")
    with st.spinner("Training LSTM Neural Network on historical data..."):
        forecast = predict_lstm(data)
    st.session_state.forecast = forecast
    valid_fc = [k for k in keywords if k in forecast.columns]
    fig_fc   = px.line(
        forecast, x=forecast.index, y=valid_fc,
        template="plotly_dark",
        title="15-Period AI Forecast (projected values)"
    )
    fig_fc.update_traces(line=dict(dash="dash"))
    st.plotly_chart(fig_fc, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ================================================================
    # SECTION 9 — WORLD HEATMAP
    # ================================================================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🌍 World Heatmap")
    region_data, err_reg = analyzer.region(keywords, geo_code)
    st.session_state.region_data = region_data

    if err_reg:
        st.error(f"Region error: {err_reg}")
    elif region_data is not None and not region_data.empty:
        rd = region_data.reset_index()
        target = keywords[0] if keywords[0] in rd.columns else rd.columns[1]
        fig_map = px.choropleth(
            rd, locations="geoName", locationmode="country names",
            color=target, template="plotly_dark",
            color_continuous_scale="Viridis",
            title=f"Global Heatmap — '{target}'"
        )
        st.plotly_chart(fig_map, use_container_width=True)

        if len(keywords) > 1:
            valid_rk = [k for k in keywords if k in rd.columns]
            rb = rd.nlargest(15, keywords[0])
            fig_rb = px.bar(
                rb, x='geoName', y=valid_rk, barmode='group',
                template='plotly_dark',
                title="Top 15 Countries — Keyword Comparison"
            )
            st.plotly_chart(fig_rb, use_container_width=True)
    else:
        st.info("No regional data available for this keyword/region combination.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ================================================================
    # SECTION 10 — RELATED QUERIES
    # ================================================================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader(f"🔍 Related Queries — '{keywords[0]}'")
    rel, err_rel = analyzer.related([keywords[0]], geo_code)
    if err_rel:
        st.warning(f"Related queries error: {err_rel}")
    elif rel and keywords[0] in rel:
        c1, c2  = st.columns(2)
        top     = rel[keywords[0]].get('top')
        rising  = rel[keywords[0]].get('rising')
        with c1:
            st.markdown("**🏆 Top Queries**")
            if top is not None and not top.empty:
                st.dataframe(top.head(10), use_container_width=True)
            else:
                st.info("No top queries found.")
        with c2:
            st.markdown("**🚀 Rising Queries**")
            if rising is not None and not rising.empty:
                st.dataframe(rising.head(10), use_container_width=True)
            else:
                st.info("No rising queries found.")
    else:
        st.info("No related query data returned.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ================================================================
    # SECTION 11 — SMART SUGGESTIONS
    # ================================================================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💡 Smart Keyword Suggestions")
    sugg, err_sugg = analyzer.suggest(keywords[0])
    if err_sugg:
        st.warning(err_sugg)
    elif sugg:
        st.dataframe(pd.DataFrame(sugg)[['title', 'type']], use_container_width=True)
    else:
        st.info("No suggestions found.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ================================================================
    # SECTION 12 — AUTO INSIGHTS
    # ================================================================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧠 AI Auto-Insights Generator")
    st.markdown(auto_insights(data, keywords))
    st.markdown('</div>', unsafe_allow_html=True)

    # ================================================================
    # SECTION 13 — EXECUTIVE SUMMARY & CASE STUDY
    # ================================================================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📝 Executive Summary & Case Study")
    st.markdown(
        f"Based on the fetched data, **{keywords[0]}** shows observable engagement across the "
        f"selected `{tf}` timeframe. The LSTM model extrapolates the next 15 periods of search "
        f"behavior. Cross-referencing the timeline with geographic heatmaps reveals regional hubs.\n\n"
        f"> **Business Case:** A company launching a product related to _{keywords[0]}_ should "
        f"focus marketing in the highest-interest regions visible on the global heatmap, and "
        f"leverage **Rising Queries** as secondary SEO keywords to capture an adjacent audience."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ================================================================
    # SECTION 14 — DOWNLOAD FULL REPORT
    # ================================================================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📥 Download Full Analytics Report")
    c1, c2 = st.columns(2)
    c1.download_button(
        "📄 Export CSV", data.to_csv().encode(), "trends.csv", "text/csv"
    )
    sheets = {"Trends": data, "Forecast": forecast}
    if region_data is not None and not region_data.empty:
        sheets["Regional"] = region_data.reset_index()
    c2.download_button(
        "📊 Export Excel Report", to_excel(sheets), "trendalyze_report.xlsx"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ================================================================
# SECTION 15 — LIVE REAL-TIME TRENDING
# ================================================================
st.markdown("---")
st.header("🔥 Live Real-Time Trending")
st.caption("Fetch the latest trending topics from Google's live search stream.")

COUNTRY_MAP = {
    "United States": "US",
    "India":         "IN",
    "United Kingdom":"GB",
    "Canada":        "CA",
    "Australia":     "AU",
}
live_country = st.selectbox("Select Region", list(COUNTRY_MAP.keys()))

if st.button("📡 Fetch Live Pulse"):
    with st.spinner("Connecting to Google's live feed..."):
        trend_df, err_t = analyzer.realtime(pn=COUNTRY_MAP[live_country])
    if err_t:
        st.error(f"Live feed error: {err_t}")
    elif not trend_df.empty:
        avail = [c for c in ['title', 'entityNames'] if c in trend_df.columns]
        st.dataframe(trend_df[avail], use_container_width=True)
    else:
        st.info("No live data returned right now. Try again in a moment.")

# ================================================================
# SECTION 16 — AI CHATBOT
# ================================================================
st.markdown("---")
st.header("🗣️ Ask the AI Analyst")
st.caption("Ask anything about your current dashboard data — trends, peaks, spikes, comparisons, forecasts.")

with st.form("chat_form", clear_on_submit=True):
    user_msg = st.text_input(
        "Your question...",
        value=st.session_state.chat_prefill,
        placeholder="e.g. Is AI growing? When was the peak? Compare keywords."
    )
    sent = st.form_submit_button("Send 💬")

if sent and user_msg:
    st.session_state.chat_prefill = ""  # Clear prefill after sending
    reply = chatbot_reply(
        user_msg,
        st.session_state.last_data,
        st.session_state.last_keywords
    )
    st.session_state.chat_history.append(("user", user_msg))
    st.session_state.chat_history.append(("bot",  reply))

st.markdown('<div class="card chat-wrap">', unsafe_allow_html=True)
if st.session_state.chat_history:
    for role, msg in st.session_state.chat_history[-14:]:
        css  = "chat-user" if role == "user" else "chat-bot"
        icon = "🧑" if role == "user" else "🤖"
        st.markdown(f'<div class="{css}">{icon} {msg}</div>', unsafe_allow_html=True)
else:
    st.caption("Generate a dashboard first, then ask the AI questions about your data.")
st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.chat_history:
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ================================================================
# FOOTER
# ================================================================
st.markdown(
    "<br><p style='text-align:center;color:gray;'>"
    "Trendalyze AI Pro · Built with Streamlit, Pytrends & TensorFlow"
    "</p>",
    unsafe_allow_html=True
)
