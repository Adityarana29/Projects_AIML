"""
💎 PhonePe Pulse — AI-Powered Advanced Dashboard
Author: Aditya Rana | M.Tech Project
"""

import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import numpy as np

# ========================================
# CONFIGURATION
# ========================================
BASE_DIR = Path.cwd()
DB_PATH = BASE_DIR / "phonepe.db"
GEOJSON_PATH = BASE_DIR / "india_states.geojson"

st.set_page_config(page_title="PhonePe AI Dashboard", page_icon="💜", layout="wide")

st.markdown("""
<style>
    .main {background-color: #0d1117; color: #f3f4f6;}
    h1,h2,h3,h4 {color: #c084fc;}
    .stMetricValue {color: #22c55e !important;}
    div[data-testid="stSidebar"] {background-color: #1f2937;}
</style>
""", unsafe_allow_html=True)

# ========================================
# HELPER FUNCTIONS
# ========================================
def load_data(query):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(query, conn)
    conn.close()
    return df

@st.cache_data(ttl=3600)
def load_geojson():
    if GEOJSON_PATH.exists():
        with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        st.warning("⚠️ GeoJSON file missing.")
        return None

def ai_summary_insights(df, state_col="State", value_col="Total_Amount"):
    """Generate simple AI-like insights heuristically."""
    top_state = df.groupby(state_col)[value_col].sum().idxmax()
    top_value = df.groupby(state_col)[value_col].sum().max()
    yearly = df.groupby("Year")[value_col].sum()
    growth = ((yearly.iloc[-1] - yearly.iloc[0]) / yearly.iloc[0]) * 100 if len(yearly) > 1 else 0
    trend = "increasing" if growth > 0 else "decreasing"
    return f"""
    • 📈 Overall transaction growth trend is **{trend} ({growth:.1f}% increase)** over the years.  
    • 🌟 **{top_state}** leads with ₹{top_value/1e9:.1f} Billion total transaction volume.  
    • 💰 The highest transaction surge occurred between **{yearly.index[-2]}–{yearly.index[-1]}**.  
    • 🧭 Continuous digital adoption indicates **stable expansion** of PhonePe usage.
    """

# ========================================
# LOAD DATA
# ========================================
df_txn = load_data("SELECT * FROM aggregated_transaction")
df_user = load_data("SELECT * FROM aggregated_user")
df_map_txn = load_data("SELECT * FROM map_transaction")

if df_txn.empty or df_user.empty:
    st.error("❌ Database empty. Please run data extraction notebook first.")
    st.stop()

df_txn.columns = [c.strip().replace(" ", "_") for c in df_txn.columns]
df_user.columns = [c.strip().replace(" ", "_") for c in df_user.columns]
df_map_txn.columns = [c.strip().replace(" ", "_") for c in df_map_txn.columns]

# ========================================
# SIDEBAR
# ========================================
st.sidebar.image("data:image/png;base64,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", width=140)
st.sidebar.title("📊 Dashboard Navigation")
page = st.sidebar.radio("Choose View", [
    "🏠 Overview",
    "💳 Transactions",
    "👥 Users",
    "🗺️ Animated Map",
    "🧠 AI Insights"
])

years = sorted(df_txn["Year"].unique())
quarters = sorted(df_txn["Quarter"].unique())
states = sorted(df_txn["State"].unique())

year_sel = st.sidebar.selectbox("Select Year", ["All"] + list(map(int, years)))
quarter_sel = st.sidebar.selectbox("Select Quarter", ["All"] + list(map(int, quarters)))
state_sel = st.sidebar.selectbox("Select State", ["All"] + states)

# Apply filters
filtered = df_txn.copy()
if year_sel != "All":
    filtered = filtered[filtered["Year"] == year_sel]
if quarter_sel != "All":
    filtered = filtered[filtered["Quarter"] == quarter_sel]
if state_sel != "All":
    filtered = filtered[filtered["State"] == state_sel]

# ========================================
# PAGE 1: OVERVIEW
# ========================================
if page == "🏠 Overview":
    st.title("💰 PhonePe Pulse — AI Analytics Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Transaction Value", f"₹{filtered['Total_Amount'].sum()/1e12:.2f}T")
    c2.metric("Transactions", f"{filtered['Transaction_Count'].sum()/1e9:.2f}B")
    c3.metric("States Covered", f"{df_txn['State'].nunique()}")

    # Yearly trend
    if {"Year", "Total_Amount"}.issubset(filtered.columns):
        trend = filtered.groupby("Year", as_index=False)["Total_Amount"].sum()
        fig = px.line(
            trend, x="Year", y="Total_Amount", markers=True,
            color_discrete_sequence=["#a855f7"], title="📈 Yearly Transaction Growth"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Type-wise split
    if "Transaction_Type" in filtered.columns:
        type_df = filtered.groupby("Transaction_Type", as_index=False)["Total_Amount"].sum()
        fig2 = px.pie(
            type_df, values="Total_Amount", names="Transaction_Type",
            title="💳 Transaction Type Share", color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig2, use_container_width=True)

# ========================================
# PAGE 2: TRANSACTIONS
# ========================================
elif page == "💳 Transactions":
    st.title("💳 Transaction Deep Dive")

    # Hotspot Bar Chart
    st.subheader("🔥 Top Transaction Hotspots (District-wise)")
    top10 = df_map_txn.groupby("District", as_index=False)["Transaction_Amount"].sum().nlargest(10, "Transaction_Amount")
    fig = px.bar(top10, x="Transaction_Amount", y="District", color="District",
                 orientation="h", title="Top 10 Districts by Transaction Volume")
    st.plotly_chart(fig, use_container_width=True)

    # Comparison over years
    st.subheader("📊 State-wise Comparison by Year")
    cmp = df_txn.groupby(["State", "Year"], as_index=False)["Total_Amount"].sum()
    fig2 = px.bar(cmp, x="Year", y="Total_Amount", color="State",
                  title="Yearly Transaction Comparison by State", barmode="group")
    st.plotly_chart(fig2, use_container_width=True)

# ========================================
# PAGE 3: USERS
# ========================================
# PAGE 3: USERS
# ========================================
elif page == "👥 Users":
    st.title("👥 User Demographics & Growth")

    # --- Detect likely user column automatically ---
    possible_user_cols = [
        c for c in df_user.columns
        if any(k in c.lower() for k in ["user", "count", "appopen"])
    ]
    if possible_user_cols:
        user_col = possible_user_cols[0]
        st.success(f"✅ Detected user column: **{user_col}**")
    else:
        st.error("❌ No user-related column found. Please check your 'aggregated_user' table.")
        st.write("Available columns:", list(df_user.columns))
        st.stop()

    # --- Ensure 'State' column exists ---
    state_col = None
    for c in df_user.columns:
        if "state" in c.lower():
            state_col = c
            break
    if state_col is None:
        st.error("❌ 'State' column missing in user data.")
        st.stop()

    # --- Top states by users ---
    top_users = (
        df_user.groupby(state_col, as_index=False)[user_col]
        .sum()
        .sort_values(user_col, ascending=False)
        .head(10)
    )

    fig = px.bar(
        top_users,
        x=user_col,
        y=state_col,
        orientation='h',
        color=user_col,
        color_continuous_scale="Purples",
        title="Top 10 States by Registered Users",
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Yearly trend of users ---
    if "Year" in df_user.columns:
        yearly_users = df_user.groupby("Year", as_index=False)[user_col].sum()
        fig2 = px.line(
            yearly_users,
            x="Year",
            y=user_col,
            markers=True,
            color_discrete_sequence=["#a855f7"],
            title="Yearly Growth in Registered Users",
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("⚠️ 'Year' column missing — cannot plot growth trend.")

# ========================================
# PAGE 4: ANIMATED MAP
# ========================================
elif page == "🗺️ Animated Map":
    st.title("🌍 Animated Growth Map (Transactions Over Years)")

    import requests

    # --- Step 1: Load India GeoJSON safely ---
    GEO_URL = "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson"
    try:
        r = requests.get(GEO_URL)
        if r.status_code == 200:
            geojson = r.json()
            st.success("✅ Loaded India GeoJSON successfully.")
        else:
            st.error("❌ GeoJSON fetch failed.")
            geojson = None
    except Exception as e:
        st.error(f"⚠️ GeoJSON load error: {e}")
        geojson = None

    if geojson:
        # --- Step 2: Detect which property key stores state names ---
        possible_keys = ["ST_NM", "NAME_1", "NAME", "state_name"]
        detected_key = None
        for key in possible_keys:
            try:
                if key in geojson["features"][0]["properties"]:
                    detected_key = key
                    break
            except Exception:
                continue

        if detected_key is None:
            st.error("❌ No recognizable state name key found in GeoJSON.")
            st.stop()

        st.caption(f"🧭 Using state name key from GeoJSON: '{detected_key}'")

        # --- Step 3: Clean state names in dataset ---
        if "State" in df_txn.columns:
            df_txn["State_Clean"] = (
                df_txn["State"]
                .astype(str)
                .str.replace("_", " ")
                .str.replace("-", " ")
                .str.strip()
                .str.title()
            )
        else:
            st.error("❌ 'State' column not found in transaction data.")
            st.stop()

        # --- Step 4: Extract valid state names from GeoJSON ---
        geo_states = [
            f["properties"][detected_key].strip().title()
            for f in geojson["features"]
            if detected_key in f["properties"]
        ]

        # --- Step 5: Fix name mismatches manually ---
        manual_map = {
            "Andaman And Nicobar Islands": "Andaman & Nicobar Islands",
            "Nct Of Delhi": "Delhi",
            "Jammu And Kashmir": "Jammu & Kashmir",
            "Arunanchal Pradesh": "Arunachal Pradesh",
            "Dadara And Nagar Havelli And Daman And Diu": "Dadra & Nagar Haveli & Daman & Diu",
        }
        df_txn["State_Clean"] = df_txn["State_Clean"].replace(manual_map)

        # --- Step 6: Check match coverage ---
        match_ratio = df_txn["State_Clean"].isin(geo_states).mean() * 100
        st.caption(f"🧩 Matching GeoJSON-State coverage: {match_ratio:.1f}%")

        # --- Step 7: Plot animated map if data is valid ---
        if match_ratio > 50 and {"State_Clean", "Year", "Total_Amount"}.issubset(df_txn.columns):
            map_df = df_txn.groupby(["State_Clean", "Year"], as_index=False)["Total_Amount"].sum()

            fig = px.choropleth(
                map_df,
                geojson=geojson,
                featureidkey=f"properties.{detected_key}",
                locations="State_Clean",
                color="Total_Amount",
                color_continuous_scale="Viridis",
                animation_frame="Year",
                title="Transaction Growth Across India (Animated by Year)",
            )
            fig.update_geos(fitbounds="locations", visible=False)
            st.plotly_chart(fig, use_container_width=True)

            st.success("✅ Map rendered successfully! Zoom and play animation to explore.")
        else:
            st.error("❌ Too few state matches or missing columns — check CSV and GeoJSON alignment.")

        # Static Choropleth for Current Year ---
        st.markdown("### 🗺️ Total Transaction Volume by State (Latest Year Snapshot)")

        latest_year = df_txn["Year"].max()
        df_latest = (
            df_txn[df_txn["Year"] == latest_year]
            .groupby("State_Clean", as_index=False)["Total_Amount"]
            .sum()
        )

        fig2 = go.Figure(
            data=go.Choropleth(
                geojson=geojson,
                featureidkey=f"properties.{detected_key}",
                locations=df_latest["State_Clean"],
                z=df_latest["Total_Amount"],
                colorscale="Purples",
                marker_line_color="white",
                colorbar=dict(
                    title={"text": "Txn Volume (₹)"},
                    thickness=15,
                    len=0.35,
                    bgcolor="rgba(255,255,255,0.6)",
                    tickfont=dict(color="black"),
                    xanchor="left",
                    x=0.01,
                    yanchor="bottom",
                    y=0.05
                )
            )
        )

        fig2.update_geos(
            visible=False,
            projection=dict(
                type="conic conformal",
                parallels=[12.4729, 35.1728],
                rotation={"lat": 24, "lon": 80}
            ),
            lonaxis={"range": [68, 98]},
            lataxis={"range": [6, 38]}
        )

        fig2.update_layout(
            title=dict(
                text=f"Total Transaction Volume by State — {latest_year}",
                xanchor="center",
                x=0.5,
                yref="paper",
                yanchor="bottom",
                y=1,
                pad={"b": 10}
            ),
            margin={"r": 0, "t": 30, "l": 0, "b": 0},
            height=550,
            width=750,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.info(f"📊 Showing total transaction volume for {latest_year}. This helps identify states with strongest UPI penetration and spending activity.")

# ========================================
# PAGE 5: AI INSIGHTS
# ========================================
elif page == "🧠 Insights":
    st.title("🧠 Business Insights Summary")

    summary = ai_summary_insights(df_txn)
    st.markdown(summary)

    # Auto anomaly detection: high variance states
    st.subheader("⚠️ High Variance Regions (Transaction Fluctuation)")
    state_var = df_txn.groupby("State")["Total_Amount"].std().sort_values(ascending=False).head(10)
    st.bar_chart(state_var)

    st.markdown("""
    ### 🧩 Business Observations:
    - States with high variance indicate volatile user behavior or emerging market transitions.
    - Steady performers reflect consistent adoption and maturity.
    - Focus areas: retention campaigns & merchant expansion in low-volume districts.
    """)

st.markdown("---")
st.caption("🚀 Built with ❤️ by Aditya Rana | M.Tech | Streamlit + Plotly + SQLite")

