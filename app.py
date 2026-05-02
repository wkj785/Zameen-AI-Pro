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

# --- 3. CUSTOM EMERALD STYLING ---
st.markdown("""
    <style>
    header {visibility: hidden;}
    .stApp { background-color: #020617; color: #ffffff; }
    
    /* Sidebar Emerald Styling */
    [data-testid="stSidebar"] { background: #0f172a; border-right: 2px solid #10b981; }
    .sidebar-brand { font-size: 2rem !important; font-weight: 900 !important; background: linear-gradient(90deg, #10b981, #ffffff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; display: block; margin-bottom: 0px; }
    .tagline { color: #10b981; font-size: 0.75rem; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; margin-top: -5px; margin-bottom: 20px; display: block; }
    
    /* Tab Styling (Predictor/History) */
    button[data-baseweb="tab"] p { color: #ffffff !important; font-size: 16px !important; font-weight: 600 !important; }
    button[data-baseweb="tab"][aria-selected="true"] { border-bottom: 3px solid #10b981 !important; background: rgba(16, 185, 129, 0.1); }

    /* Input Emerald Styling (Location, Area, Beds, etc.) */
    label[data-testid="stWidgetLabel"] p { color: #10b981 !important; font-weight: 800 !important; text-transform: uppercase; font-size: 0.85rem; }
    div[data-baseweb="select"], div[data-baseweb="input"], div[data-baseweb="base-input"] {
        border: 1px solid #10b981 !important; border-radius: 8px !important; background-color: #0f172a !important;
    }
    
    /* Emerald Button */
    div.stButton > button { 
        background-color: transparent !important; color: #10b981 !important; 
        border: 2px solid #10b981 !important; border-radius: 8px; font-weight: 800 !important; 
        width: 100% !important; transition: 0.3s;
    }
    div.stButton > button:hover { background-color: #10b981 !important; color: #020617 !important; box-shadow: 0 0 15px #10b981; }
    
    /* Cards */
    .module-card { background-color: #0f172a; padding: 20px; border-radius: 12px; border: 1px solid #10b981; margin-bottom: 15px; }
    .metric-val { color: #10b981; font-size: 1.5rem; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. AUTHENTICATION (Skip if already logged in) ---
if 'auth_status' not in st.session_state:
    st.session_state.auth_status = False

# ... (Insert Auth Logic here if needed, keeping simple for this update) ...

# --- 5. SIDEBAR BRANDING ---
with st.sidebar:
    st.markdown('<p class="sidebar-brand">Zameen AI Pro</p>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">Hybrid Intelligence Valuation</p>', unsafe_allow_html=True)
    st.write(f"Logged in: **{st.session_state.get('username', 'WKJ.785')}**")
    st.divider()
    
    st.markdown("### 📏 Area Converter")
    conv_val = st.number_input("Enter Value", value=1.0)
    conv_type = st.selectbox("Convert From", ["Marlas to SqYd", "Kanals to SqYd", "SqYd to Marlas"])
    
    if conv_type == "Marlas to SqYd":
        res = conv_val * 30.25
        st.info(f"{conv_val} Marla = {res:.2f} SqYd")
    elif conv_type == "Kanals to SqYd":
        res = conv_val * 605
        st.info(f"{conv_val} Kanal = {res:.2f} SqYd")
    
    if st.button("🚪 LOGOUT"):
        st.session_state.auth_status = False
        st.rerun()

# --- 6. MAIN DASHBOARD ---
main_tab, hist_tab = st.tabs(["🚀 Predictor", "📜 History"])

with main_tab:
    # Top Row: Analytics & Sentiment
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        st.markdown('<div class="module-card">', unsafe_allow_html=True)
        st.subheader("📍 Live Market Map")
        # Placeholder for Map - using a colored area for visual structure
        st.map(pd.DataFrame({'lat': [33.6844], 'lon': [73.0479]})) # Islamabad Center
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_b:
        st.markdown('<div class="module-card">', unsafe_allow_html=True)
        st.subheader("📊 Market Sentiment")
        st.write("Current Trend: **Bullish**")
        st.progress(85)
        st.caption("85% Positive market activity in this sector.")
        st.markdown('<hr style="border-color:#10b981;">', unsafe_allow_html=True)
        st.write("Volatility: **Low**")
        st.markdown('</div>', unsafe_allow_html=True)

    # Middle Row: Predictor Inputs
    st.markdown('<div class="module-card">', unsafe_allow_html=True)
    loc_name = st.selectbox("Location / Sector", ["AGHOSH, Islamabad", "DHA Phase 6, Lahore", "Bahria Town, Karachi"])
    
    c1, c2, c3, c4 = st.columns(4)
    area = c1.number_input("Area (SqYd)", 1, 5000, 125)
    beds = c2.number_input("Beds", 1, 10, 3)
    baths = c3.number_input("Baths", 1, 10, 3)
    kitchens = c4.number_input("Kitchens", 1, 5, 1)
    
    if st.button("🚀 GENERATE HYBRID VALUATION"):
        st.success("Analysis Complete!")
        # Price Calculation Logic would go here
    st.markdown('</div>', unsafe_allow_html=True)

with hist_tab:
    st.markdown('<div class="module-card">', unsafe_allow_html=True)
    st.subheader("Property Valuation History")
    # Placeholder Dataframe
    history_data = pd.DataFrame({
        'Date': ['2026-05-02', '2026-05-01'],
        'Location': ['AGHOSH', 'DHA Phase 6'],
        'Price (PKR)': ['4.5 Crore', '12.2 Crore']
    })
    st.dataframe(history_data, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
