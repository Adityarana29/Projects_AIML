import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ------------------------
# CONFIG
# ------------------------
MODEL_DIR = "saved_models"
CLASSIFIER_PATH = f"{MODEL_DIR}/best_classifier.pkl"
REGRESSOR_PATH = f"{MODEL_DIR}/best_regressor.pkl"
CLEANED_DATA = "cleaned_india_housing_prices.csv"   # your dataset

# Load models
clf = joblib.load(CLASSIFIER_PATH)
reg = joblib.load(REGRESSOR_PATH)

# App settings
st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏙️", layout="wide")

# Title
st.markdown("<h1 style='text-align:center;'>Real Estate Investment Advisor 🏙️</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-based Investment Classification & Projection</p>", unsafe_allow_html=True)
st.markdown("---")

# Load dataset for insights
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(CLEANED_DATA)
        st.success(f"Loaded dataset: {CLEANED_DATA}")
        return df
    except Exception as e:
        st.error(f"❌ Could not load dataset `{CLEANED_DATA}`: {e}")
        return None

df = load_data()

# -------------------------------------
# USER INPUT (Simplified)
# -------------------------------------
st.header("📥 Enter Property Information")

col1, col2, col3 = st.columns(3)

with col1:
    bhk = st.number_input("BHK", 1, 10, 2)
    size = st.number_input("Size (SqFt)", 200, 10000, 1200)
    price_psqft = st.number_input("Price per SqFt (₹)", 20, 20000, 5000)

with col2:
    year_built = st.number_input("Year Built", 1950, 2025, 2015)
    nearby_schools = st.number_input("Nearby Schools", 0, 20, 3)
    nearby_hospitals = st.number_input("Nearby Hospitals", 0, 20, 2)
    pta = st.number_input("Public Transport Accessibility (0–5)", 0, 5, 3)

with col3:
    parking = st.number_input("Parking Spaces", 0, 5, 1)
    floor_no = st.number_input("Floor", 0, 50, 1)
    total_floors = st.number_input("Total Floors", 1, 50, 5)

# Derived features
age_of_property = 2025 - year_built
price_lakhs = (price_psqft * size) / 100000

# Minimal categorical defaults (because your model expects them)
categorical_defaults = {
    "State": "Unknown",
    "City": "Unknown",
    "Locality": "Unknown",
    "Property_Type": "Apartment",
    "Furnished_Status": "Unknown",
    "Security": "Medium",
    "Facing": "East",
    "Owner_Type": "Individual",
    "Availability_Status": "Ready",
    "Amenity_Count": 2
}

# Build dataframe for model
input_data = pd.DataFrame([{
    "BHK": bhk,
    "Size_in_SqFt": size,
    "Price_per_SqFt": price_psqft,
    "Price_in_Lakhs": price_lakhs,
    "Year_Built": year_built,
    "Nearby_Schools": nearby_schools,
    "Nearby_Hospitals": nearby_hospitals,
    "Parking_Space": parking,
    "Floor_No": floor_no,
    "Total_Floors": total_floors,
    "Public_Transport_Accessibility": pta,   # ✔ FIXED & ADDED
    "Age_of_Property": age_of_property,
    **categorical_defaults
}])

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Prediction & Radar Score", "📊 Compare Your Property", "🏠 Property Search", "📈 Market Analytics"])

# ---------------------------------------------------------------------------
# TAB 1 — PREDICTION + RADAR SCORE
# ---------------------------------------------------------------------------
with tab1:

    st.subheader("🔍 Prediction Result")

    if st.button("Predict Investment Value"):
        pred_class = clf.predict(input_data)[0]
        predicted_5yr_price = reg.predict(input_data)[0] * 100000

        colA, colB = st.columns(2)

        with colA:
            st.subheader("Investment Decision")
            if pred_class == 1:
                st.success("GOOD INVESTMENT ✅")
            else:
                st.error("NOT A GOOD INVESTMENT ❌")

        with colB:
            st.subheader("Estimated 5-Year Price")
            st.metric("Price Projection", f"₹ {predicted_5yr_price:,.0f}")

        st.markdown("---")

        # ---------------------
        # RADAR SCORE
        # ---------------------
        st.subheader("📌 Investment Score Radar Chart")

        radar_scores = {
            "Price Efficiency": min(100, (8000 / price_psqft) * 50),
            "Size Score": min(100, size / 150),
            "BHK Score": bhk * 10,
            "Age Score": max(0, 100 - age_of_property * 2),
            "Amenities Score": (nearby_schools + nearby_hospitals) * 5,
            "Transport Score": pta * 20,
            "Parking Score": parking * 20
        }

        categories = list(radar_scores.keys())
        values = list(radar_scores.values())
        values.append(values[0])  # close chart

        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill="toself",
            line=dict(color="cyan")
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2 — COMPARE PROPERTY VS MARKET
# ---------------------------------------------------------------------------
with tab2:

    st.subheader("📊 Compare Your Property With Market Trends")

    if df is None:
        st.warning("Dataset missing — comparison unavailable.")
    else:

        # Market medians
        market_price = df["Price_per_SqFt"].median()
        market_size = df["Size_in_SqFt"].median()
        market_bhk = df["BHK"].median()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Your PPSF", price_psqft)
            st.metric("Market PPSF Median", market_price)

        with col2:
            st.metric("Your Size", size)
            st.metric("Market Median Size", market_size)

        with col3:
            st.metric("Your BHK", bhk)
            st.metric("Market Median BHK", market_bhk)

        st.markdown("---")

        # Comparison chart
        compare_df = pd.DataFrame({
            "Feature": ["Price per SqFt", "Size (SqFt)", "BHK"],
            "Your Property": [price_psqft, size, bhk],
            "Market Median": [market_price, market_size, market_bhk]
        })

        fig_compare = px.bar(
            compare_df,
            x="Feature",
            y=["Your Property", "Market Median"],
            barmode="group",
            title="Your Property vs Market Comparison",
            text_auto=True
        )

        st.plotly_chart(fig_compare, use_container_width=True)

        st.markdown("---")

        st.subheader("📌 Insight Summary")

        if price_psqft < market_price:
            st.success("Your PPSF is **below market average** → Good value!")
        else:
            st.warning("Your PPSF is **above market median**.")

        if size > market_size:
            st.success("Your property is **larger than average**.")
        else:
            st.info("Property size is **below market average**.")

        if bhk >= market_bhk:
            st.success("Your BHK is **competitive**.")
        else:
            st.warning("Your BHK is **below market median**.")

# ---------------------------------------------------------------------------
# TAB 3 — PROPERTY SEARCH
# ---------------------------------------------------------------------------
with tab3:

    st.subheader("🏠 Property Search Engine")

    if df is None:
        st.warning("Dataset missing — search unavailable.")
    else:
        # City Filter
        city_list = sorted(df["City"].dropna().unique())
        selected_city = st.selectbox("Select City", city_list)

        # Filter localities based on selected city
        locality_list = sorted(df[df["City"] == selected_city]["Locality"].dropna().unique())
        selected_locality = st.selectbox("Select Locality", locality_list)

        # Additional Filters
        colA, colB, colC = st.columns(3)

        with colA:
            min_bhk, max_bhk = st.slider("BHK Range", 1, 10, (1, 5))

        with colB:
            min_price, max_price = st.slider("Price in Lakhs", 5, 20000, (50, 500))

        with colC:
            min_size, max_size = st.slider("Size (SqFt)", 200, 10000, (500, 4000))

        # Apply Filters
        result_df = df[
            (df["City"] == selected_city) &
            (df["Locality"] == selected_locality) &
            (df["BHK"].between(min_bhk, max_bhk)) &
            (df["Price_in_Lakhs"].between(min_price, max_price)) &
            (df["Size_in_SqFt"].between(min_size, max_size))
        ]

        st.markdown("### 🔎 Search Results")
        st.dataframe(result_df, use_container_width=True)

        if len(result_df) > 0:
            st.markdown("### 📊 Price vs Size (Filtered Results)")
            fig_search = px.scatter(
                result_df,
                x="Size_in_SqFt",
                y="Price_in_Lakhs",
                color="BHK",
                size="Price_in_Lakhs",
                title=f"Filtered Properties in {selected_locality}, {selected_city}"
            )
            st.plotly_chart(fig_search, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 4 — MARKET ANALYTICS
# ---------------------------------------------------------------------------
with tab4:

    st.subheader("📈 Market Analytics")

    if df is None:
        st.warning("Dataset missing — analytics unavailable.")
    else:
        # ------------------------------
        # 1. Average Price per SqFt by City
        # ------------------------------
        st.markdown("### 🏙️ Average Price per SqFt by City")

        city_ppsf = df.groupby("City")["Price_per_SqFt"].mean().sort_values(ascending=False)

        fig_city_ppsf = px.bar(
            city_ppsf,
            title="Average Price per SqFt by City",
            labels={"value": "Price per SqFt"},
            color=city_ppsf.values
        )
        st.plotly_chart(fig_city_ppsf, use_container_width=True)

        # ------------------------------
        # 2. Year-over-Year Growth Rate
        # ------------------------------
        if "Year_Built" in df.columns:

            st.markdown("### 📆 Year-over-Year Property Growth Trend")

            yoy = df.groupby("Year_Built")["Price_per_SqFt"].mean().pct_change() * 100
            yoy_df = yoy.reset_index()
            yoy_df.columns = ["Year", "YoY_Growth"]

            fig_yoy = px.line(
                yoy_df,
                x="Year",
                y="YoY_Growth",
                markers=True,
                title="Year-over-Year Growth Rate (%)"
            )
            st.plotly_chart(fig_yoy, use_container_width=True)

            st.info("YoY Growth is based on the trend of Price per SqFt across property construction years.")

        else:
            st.warning("❗ 'Year_Built' column missing — cannot compute YoY Growth.")

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("Developed for M.Tech Project — Real Estate Investment Advisor")
