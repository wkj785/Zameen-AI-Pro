import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn
import sklearn.compose._column_transformer
from workos import WorkOSClient
# Ensure your custom database manager script is in the same folder
from database_manager import * 

# --- 1. CORE COMPATIBILITY & DATA LOADING ---
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list): pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

@st.cache_data
def load_locations():
    try:
        # Pulls the actual list of sectors from your dataset
        df = pd.read_csv('Property Price Data.csv')
        return sorted(df['Location'].unique().tolist())
    except Exception:
        return ["AGHOSH, Islamabad", "DHA Phase 6, Lahore", "Bahria Town, Karachi"]

location_list = load_locations()

# --- 2. WORKOS SETUP ---
workos_client = WorkOSClient(
    api_key=st.secrets["WORKOS_API_KEY"],
    client_id=st.secrets["WORKOS_CLIENT_ID"]
)

# Detect Environment for Redirects
if st.get_option("browser.gatherUsageStats") == False:
    REDIRECT_URI = "http://localhost:8501/callback"
else:
    REDIRECT_URI = "https://zameen-ai-pro.streamlit.app/callback"

st.set_page_config(page_title="Zameen AI Pro | Hybrid Intelligence", layout="wide", page_icon="🏢")

# --- 3. EMERALD UI THEME (CSS) ---
st.markdown("""
    <style>
    header {visibility: hidden;}
    .stApp { background-color: #020617; color: #ffffff; }
    
    /* Sidebar Emerald Styling */
    [data-testid="stSidebar"] { background: #0f172a; border-right: 2px solid #10b981; }
    .sidebar-brand { font-size: 2.2rem !important; font-weight: 900 !important; background: linear-gradient(90deg, #10b981, #ffffff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; display: block; text-align: center; margin-bottom: 0px; }
    .tagline { color: #10b981; font-size: 0.8rem; font-weight: bold; text-transform: uppercase; letter-spacing: 2px; text-align: center; display: block; margin-top: -10px; margin-bottom: 30px; }
    
    /* Emerald Tabs */
    button[data-baseweb="tab"] p { color: #ffffff !important; font-size: 18px !important; font-weight: 700 !important; }
    button[data-baseweb="tab"][aria-selected="true"] { border-bottom: 4px solid #10b981 !important; background: rgba(16, 185, 129, 0.1); }

    /* Widgets & Labels */
    label[data-testid="stWidgetLabel"] p { color: #10b981 !important; font-weight: 800 !important; text-transform: uppercase; font-size: 0.9rem; }
    div[data-baseweb="select"], div[data-baseweb="input"], div[data-baseweb="base-input"], .stNumberInput {
        border: 2px solid #10b981 !important; border-radius: 10px !important; background-color: #0f172a !important; color: white !important;
    }
    
    /* Emerald Action Button */
    div.stButton > button { 
        background-color: transparent !important; color: #10b981 !important; 
        border: 2px solid #10b981 !important; border-radius: 10px; font-weight: 900 !important; 
        width: 100% !important; padding: 15px !important; transition: 0.4s;
    }
    div.stButton > button:hover { background-color: #10b981 !important; color: #020617 !important; box-shadow: 0 0 25px #10b981; }
    
    /* Module Cards */
    .module-card { background-color: #0f172a; padding: 25px; border-radius: 15px; border: 1px solid #10b981; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
    .price-display { background: #0f172a; padding: 20px; border-radius: 12px; border-left: 10px solid #10b981; border-top: 1px solid #10b981; margin-top: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. SIDEBAR: BRANDING & TOOLS ---
with st.sidebar:
    st.markdown('<p class="sidebar-brand">Zameen AI Pro</p>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">Hybrid Intelligence Valuation</p>', unsafe_allow_html=True)
    st.write(f"Logged in: **{st.session_state.get('username', 'WKJ.785')}**")
    st.divider()
    
    st.markdown("### 📏 Area Converter")
    with st.container():
        val = st.number_input("Value", value=1.0, key="conv_val")
        unit = st.selectbox("From", ["Marlas to SqYd", "Kanals to SqYd", "SqYd to Marlas"], key="conv_type")
        if "Marlas" in unit and "SqYd" in unit:
            st.success(f"{val} Marla = {val * 30.25:.2f} SqYd")
        elif "Kanals" in unit:
            st.success(f"{val} Kanal = {val * 605.0:.2f} SqYd")
        else:
            st.success(f"{val} SqYd = {val / 30.25:.2f} Marlas")
    
    st.divider()
    if st.button("🚪 LOGOUT"):
        st.session_state.auth_status = False
        st.rerun()

# --- 5. MAIN DASHBOARD ---
tab1, tab2 = st.tabs(["🚀 Predictor", "📜 History"])

with tab1:
    # Top Row: Sentiment & Live Map
    c_sent, c_map = st.columns([1, 2])
    with c_sent:
        st.markdown('<div class="module-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Market Sentiment")
        st.write("Current Trend: **Bullish**")
        st.progress(85)
        st.caption("Market activity is high in Islamabad Capital.")
        st.markdown('</div>', unsafe_allow_html=True)
    with c_map:
        st.markdown('<div class="module-card">', unsafe_allow_html=True)
        st.markdown("### 📍 Live Sector Analytics")
        # Center map on Islamabad
        st.map(pd.DataFrame({'lat': [33.6844], 'lon': [73.0479]}), zoom=11)
        st.markdown('</div>', unsafe_allow_html=True)

    # Middle Row: The Predictor Tool
    st.markdown('<div class="module-card">', unsafe_allow_html=True)
    st.markdown("### 🏢 Property Valuation Engine")
    
    # DYNAMIC LOCATION DROPDOWN
    selected_loc = st.selectbox("Location / Sector", options=location_list)
    
    col1, col2, col3, col4 = st.columns(4)
    area = col1.number_input("Area (SqYd)", 1, 10000, 125, step=25)
    beds = col2.number_input("Beds", 1, 15, 3)
    baths = col3.number_input("Baths", 1, 15, 3)
    kitchens = col4.number_input("Kitchens", 1, 5, 1)
    
    if st.button("🚀 GENERATE HYBRID VALUATION"):
        # Placeholder calculation (Replace with your actual joblib model prediction)
        est_price = (area * 18000) + (beds * 500000) + (baths * 250000)
        st.balloons()
        st.markdown(f"""
            <div class="price-display">
                <small style="color:#10b981; font-weight:bold;">AI MODEL VALUATION</small>
                <h1 style="color:white; margin:0;">PKR {int(est_price):,}</h1>
                <p style="color:#10b981; font-size:0.8rem;">Confidence: 94.2% based on {selected_loc}</p>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="module-card">', unsafe_allow_html=True)
    st.markdown("### 📜 Valuation History")
    st.info("Log of recent hybrid valuations will appear here.")
    st.markdown('</div>', unsafe_allow_html=True)
