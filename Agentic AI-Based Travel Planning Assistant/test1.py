import streamlit as st
import json
import pandas as pd
import requests
import folium
from streamlit_folium import st_folium
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import tempfile

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config("Smart Travel Planner", "🌍", layout="wide")

# --------------------------------------------------
# Session State
# --------------------------------------------------
if "generate_clicked" not in st.session_state:
    st.session_state.generate_clicked = False

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

places_df = pd.DataFrame(load_json("data/places.json"))
hotels_df = pd.DataFrame(load_json("data/hotels.json"))
flights_df = pd.DataFrame(load_json("data/flights.json"))

# --------------------------------------------------
# Sidebar – Controls
# --------------------------------------------------
st.sidebar.title("⚙️ Preferences")

accent_color = st.sidebar.selectbox(
    "🎨 Accent Color",
    ["#1f3c88", "#0d6efd", "#198754", "#6f42c1"]
)

dark_mode = st.sidebar.toggle("🌙 Dark Mode")
agent_mode = st.sidebar.radio("🧠 Planner Mode", ["Rule-Based", "Agentic (Future)"])

# --------------------------------------------------
# Dynamic CSS
# --------------------------------------------------
bg = "#0e1117" if dark_mode else "#ffffff"
fg = "#eaeaea" if dark_mode else "#000000"

st.markdown(f"""
<style>
.hero {{
    font-size: 36px;
    font-weight: 700;
    color: {accent_color};
}}

.hero-sub {{
    font-size: 18px;
    color: #6c757d;
    margin-bottom: 20px;
}}

.section-header {{
    font-size: 22px;
    font-weight: 600;
    margin-top: 20px;
    border-bottom: 3px solid {accent_color};
    padding-bottom: 6px;
}}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Constants
# --------------------------------------------------
CITY_COORDS = {
    "Delhi": (28.61, 77.23),
    "Mumbai": (19.07, 72.87),
    "Goa": (15.49, 73.82),
    "Bangalore": (12.97, 77.59),
    "Chennai": (13.08, 80.27),
    "Hyderabad": (17.38, 78.48),
    "Kolkata": (22.57, 88.36),
    "Jaipur": (26.91, 75.79)
}

TYPE_DESC = {
    "beach": "Popular coastal attraction ideal for relaxation.",
    "temple": "Spiritual landmark of cultural importance.",
    "fort": "Historic fort reflecting royal architecture.",
    "museum": "Museum preserving art and history.",
    "park": "Green space suitable for leisure.",
    "market": "Local shopping and food destination.",
    "lake": "Scenic lake with calm surroundings.",
    "monument": "Iconic heritage structure."
}

# --------------------------------------------------
# Weather
# --------------------------------------------------
def get_weather(city, days):
    lat, lon = CITY_COORDS[city]
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max&timezone=auto"
    return requests.get(url, timeout=10).json()["daily"]["temperature_2m_max"][:days]

# --------------------------------------------------
# Sidebar – Trip Inputs
# --------------------------------------------------
st.sidebar.title("🧳 Trip Inputs")

from_city = st.sidebar.selectbox("✈️ From", sorted(flights_df["from"].unique()))
destination = st.sidebar.selectbox("📍 To", sorted(places_df["city"].unique()))
days = st.sidebar.slider("🗓 Trip Duration (Days)", 1, 7, 3)
price_limit = st.sidebar.slider("💰 Max Flight Price", 2000, 10000, 6000)

if st.sidebar.button("✨ Generate Travel Plan"):
    st.session_state.generate_clicked = True

# --------------------------------------------------
# Main
# --------------------------------------------------
if st.session_state.generate_clicked:

    # ---------------- Flights ----------------
    flights = flights_df[
        (flights_df["from"] == from_city) &
        (flights_df["to"] == destination) &
        (flights_df["price"] <= price_limit)
    ]
    flight_cost = flights["price"].min() if not flights.empty else 0

    # ---------------- Hotels ----------------
    city_hotels = hotels_df[hotels_df["city"] == destination]

    best_hotel = None
    if not city_hotels.empty:
        best_hotel = city_hotels.sort_values(
            by=["stars", "price_per_night"],
            ascending=[False, True]
        ).iloc[0]

    hotel_cost = days * (best_hotel["price_per_night"] if best_hotel is not None else 3000)
    local_cost = days * 1500
    total_cost = flight_cost + hotel_cost + local_cost
    st.markdown(
    f"""
    <h1 style="text-align:center;">🌍 {destination} Travel Planner</h1>""",
    unsafe_allow_html=True
    )
    # ---------------- Tabs ----------------
    tab_home, tab_weather, tab_itinerary, tab_map, tab_budget, tab_export = st.tabs(
        ["🏠 Home", "🌦 Weather", "🗓 Itinerary", "🗺 Map", "💰 Budget", "📄 Export"]
    )

    # ================= HOME =================
    with tab_home:
        st.markdown(f"""
        <div class="hero">✈️ {from_city} → 🌍 {destination}</div>
        <div class="hero-sub">{days}-Day Trip • Estimated Budget ₹{total_cost}</div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        # Flight
        with col1:
            st.markdown("### ✈️ Flight")
            if not flights.empty:
                f = flights.sort_values("price").iloc[0]
                st.write(f"**Airline:** {f['airline']}")
                st.write(f"**Departure:** {f['departure_time']}")
                st.write(f"**Price:** ₹{f['price']}")
            else:
                st.warning("No flight within budget")

        # Hotel
        with col2:
            st.markdown("### 🏨 Hotel")
            if best_hotel is not None:
                st.write(f"**Name:** {best_hotel['name']}")
                st.write(f"**Rating:** ⭐ {best_hotel['stars']}")
                st.write(f"**Price/Night:** ₹{best_hotel['price_per_night']}")
            else:
                st.warning("No hotels available")

        # Map
        with col3:
            st.markdown("### 🗺 Location")
            lat, lon = CITY_COORDS[destination]
            m = folium.Map(location=[lat, lon], zoom_start=10)
            folium.Marker([lat, lon], popup=destination).add_to(m)
            st_folium(m, width=350, height=250)

    # ================= WEATHER =================
    with tab_weather:
        for i, t in enumerate(get_weather(destination, days), 1):
            st.write(f"Day {i}: 🌡 {t}°C")

    # ================= ITINERARY =================
    with tab_itinerary:
        places = places_df[places_df["city"] == destination].sort_values("rating", ascending=False)
        idx = 0
        for d in range(1, days + 1):
            day_places = places.iloc[idx:idx + 2]
            idx += 2
            if day_places.empty:
                break
            st.markdown(f"### Day {d}")
            for _, p in day_places.iterrows():
                st.markdown(f"📍 **{p['name']}**  \n{TYPE_DESC.get(p['type'])}")

    # ================= MAP =================
    with tab_map:
        lat, lon = CITY_COORDS[destination]
        m = folium.Map(location=[lat, lon], zoom_start=11)
        folium.Marker([lat, lon], popup=destination).add_to(m)
        st_folium(m, width=1000, height=450)

    # ================= BUDGET =================
    with tab_budget:
        budget_df = pd.DataFrame({
            "Category": ["Flight", "Hotel", "Local"],
            "Cost (₹)": [flight_cost, hotel_cost, local_cost]
        })
        st.bar_chart(budget_df.set_index("Category"))
        st.success(f"Total Estimated Cost: ₹{total_cost}")


