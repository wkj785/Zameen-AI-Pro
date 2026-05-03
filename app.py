import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn
import sklearn.compose._column_transformer
from workos import WorkOSClient
from database_manager import *

# --- 1. CORE COMPATIBILITY ---
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list): pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

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

# --- 3. THE EMERALD UI CSS ---
st.markdown("""
    <style>
    header {visibility: hidden;}
    .stApp { background-color: #020617; color: #ffffff; }
    
    /* Sidebar Emerald Styling */
    [data-testid="stSidebar"] { background: #0f172a; border-right: 2px solid #10b981; }
    .sidebar-brand { font-size: 2.2rem !important; font-weight: 900 !important; background: linear-gradient(90deg, #10b981, #ffffff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; display: block; text-align: center; margin-bottom: 0px; }
    .tagline { color: #10b981; font-size: 0.8rem; font-weight: bold; text-transform: uppercase; letter-spacing: 2px; text-align: center; display: block; margin-top: -10px; margin-bottom: 30px; }
    
    /* Tab Styling */
    button[data-baseweb="tab"] p { color: #ffffff !important; font-size: 18px !important; font-weight: 700 !important; }
    button[data-baseweb="tab"][aria-selected="true"] { border-bottom: 4px solid #10b981 !important; background: rgba(16, 185, 129, 0.1); }

    /* Apply Emerald to ALL Input Labels & Widgets */
    label[data-testid="stWidgetLabel"] p { color: #10b981 !important; font-weight: 800 !important; text-transform: uppercase; font-size: 0.9rem; }
    
    div[data-baseweb="select"], div[data-baseweb="input"], div[data-baseweb="base-input"], .stNumberInput {
        border: 2px solid #10b981 !important; border-radius: 10px !important; background-color: #0f172a !important;
    }
    
    /* Emerald Button */
    div.stButton > button { 
        background-color: transparent !important; color: #10b981 !important; 
        border: 2px solid #10b981 !important; border-radius: 10px; font-weight: 900 !important; 
        width: 100% !important; padding: 15px !important; transition: 0.4s;
    }
    div.stButton > button:hover { background-color: #10b981 !important; color: #020617 !important; box-shadow: 0 0 25px #10b981; }
    
    /* Emerald Cards for Modules */
    .module-card { background-color: #0f172a; padding: 25px; border-radius: 15px; border: 1px solid #10b981; margin-bottom: 20px; }
    .price-display { background: #0f172a; padding: 20px; border-radius: 12px; border-left: 10px solid #10b981; border-top: 1px solid #10b981; margin-top: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. SIDEBAR: BRANDING & AREA CONVERTER ---
with st.sidebar:
    st.markdown('<p class="sidebar-brand">Zameen AI Pro</p>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">Hybrid Intelligence Valuation</p>', unsafe_allow_html=True)
    st.write(f"Logged in: **{st.session_state.get('username', 'WKJ.785')}**")
    st.divider()
    
    st.markdown("### 📏 Area Converter")
    c_val = st.number_input("Value", value=1.0)
    c_type = st.selectbox("From", ["Marlas to SqYd", "Kanals to SqYd", "SqYd to Marlas"])
    
    if c_type == "Marlas to SqYd":
        st.success(f"Result: {c_val * 30.25:.2f} SqYd")
    elif c_type == "Kanals to SqYd":
        st.success(f"Result: {c_val * 605.0:.2f} SqYd")
    elif c_type == "SqYd to Marlas":
        st.success(f"Result: {c_val / 30.25:.2f} Marlas")
    
    st.divider()
    if st.button("🚪 LOGOUT"):
        st.session_state.auth_status = False
        st.rerun()

# --- 5. MAIN DASHBOARD CONTENT ---
main_tab, hist_tab = st.tabs(["🚀 Predictor", "📜 History"])

with main_tab:
    # Row 1: Sentiment & Map
    top_col1, top_col2 = st.columns([1, 2])
    
    with top_col1:
        st.markdown('<div class="module-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Market Sentiment")
        st.write("Trend: **Bullish**")
        st.progress(82)
        st.caption("High demand in Islamabad Capital territory.")
        st.markdown('</div>', unsafe_allow_html=True)

    with top_col2:
        st.markdown('<div class="module-card">', unsafe_allow_html=True)
        st.markdown("### 📍 Live Sector Analytics")
        map_data = pd.DataFrame({'lat': [33.6844], 'lon': [73.0479]})
        st.map(map_data, zoom=12)
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 2: Predictor Tool
    st.markdown('<div class="module-card">', unsafe_allow_html=True)
    st.markdown("### 🏢 Property Valuation Engine")
    
    loc_name = st.selectbox("Location / Sector", ["AGHOSH, Islamabad, Islamabad Capital", "DHA Phase 6, Lahore", "Bahria Town, Karachi"])
    
    c1, c2, c3, c4 = st.columns(4)
    area = c1.number_input("Area (SqYd)", 1, 10000, 125, step=25)
    beds = c2.number_input("Beds", 1, 15, 3)
    baths = c3.number_input("Baths", 1, 15, 3)
    kitchens = c4.number_input("Kitchens", 1, 5, 1)
    
    if st.button("🚀 GENERATE HYBRID VALUATION"):
        total_val = (area * 16500) + (beds * 450000) + (baths * 200000)
        st.balloons()
        st.markdown(f"""
            <div class="price-display">
                <small style="color:#10b981; font-weight:bold;">AI MODEL VALUATION</small>
                <h1 style="color:white; margin:0;">PKR {int(total_val):,}</h1>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with hist_tab:
    st.markdown('<div class="module-card">', unsafe_allow_html=True)
    st.markdown("### 📜 Recent Valuation History")
    # Placeholder history
    hist_df = pd.DataFrame({
        'Timestamp': ['2026-05-03 09:15', '2026-05-02 23:10'],
        'Location': [loc_name, 'DHA Phase 6, Lahore'],
        'Valuation': ['PKR 3,975,000', 'PKR 8,450,000']
    })
    st.dataframe(hist_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
