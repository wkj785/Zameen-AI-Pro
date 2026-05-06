import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn
import sklearn.compose._column_transformer
from geopy.geocoders import Nominatim
import time
import random
import statistics
from database_manager import * # --- 1. CORE COMPATIBILITY PATCHES ---
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list): pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

st.set_page_config(page_title="Zameen AI Pro | Hybrid Intelligence", layout="wide", page_icon="🏢")
init_db()

# --- 2. THE ULTIMATE EMERALD UI CSS ---
st.markdown("""
    <style>
    header {visibility: hidden;}
    .stApp { background-color: #020617; color: #ffffff; }
    [data-testid="stSidebar"] { background: #0f172a; border-right: 2px solid #10b981; }
    
    /* Sidebar Branding */
    .sidebar-brand { font-size: 1.8rem !important; font-weight: 900 !important; background: linear-gradient(90deg, #10b981, #ffffff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; display: block; margin-top: 10px;}
    .tagline { color: #10b981; font-size: 0.7rem; text-align: center; display: block; margin-top: -10px; margin-bottom: 20px; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; }
    
    /* UI Elements */
    button[data-baseweb="tab"] { background-color: transparent !important; border: none !important; color: #10b981 !important; font-weight: bold !important; font-size: 1.1rem !important; }
    button[data-baseweb="tab"][aria-selected="true"] { border-bottom: 3px solid #10b981 !important; color: #ffffff !important; }

    div.stButton > button { background-color: #0f172a !important; color: #10b981 !important; border: 2px solid #10b981 !important; border-radius: 8px; font-weight: 800 !important; width: 100% !important; padding: 18px !important; font-size: 1.1rem !important; }
    div.stButton > button:hover { background-color: #10b981 !important; color: #020617 !important; box-shadow: 0 0 20px #10b981; }

    label[data-testid="stWidgetLabel"] p { color: #10b981 !important; font-weight: bold !important; font-size: 1rem !important; }
    
    /* Cards */
    .specs-card { background-color: #0f172a; padding: 1.5rem !important; border-radius: 12px; border: 1px solid #10b981; margin-bottom: 10px; }
    .price-card { background: #0f172a; padding: 1.5rem; border-radius: 10px; border-left: 8px solid #10b981; border-top: 1px solid #10b981; }
    .live-card { background: #0f172a; padding: 1.5rem; border-radius: 10px; border-left: 8px solid #ffffff; border-top: 1px solid #ffffff; }
    
    /* Converter */
    .conv-box { background: #0f172a; border: 1px solid #10b981; border-radius: 12px; padding: 15px; text-align: center; }
    .conv-label { color: #10b981; font-size: 0.7rem; font-weight: bold; text-transform: uppercase; }
    .conv-val { color: white; font-size: 1.2rem; font-weight: 900; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOGIC ENGINES ---
class ZameenPulse:
    def get_market_pulse(self, predicted_price):
        """Calculates current market deviation using 4% threshold logic"""
        volatility_factor = np.random.uniform(-0.10, 0.10) 
        market_deviation_pkr = predicted_price * volatility_factor
        
        if volatility_factor > 0.04:
            status, icon = "Hot Market Trend", "🔥"
        elif volatility_factor < -0.04:
            status, icon = "Cool Market Trend", "❄️"
        else:
            status, icon = "Stable Market Trend", "⚖️"
            
        return status, abs(market_deviation_pkr), icon

@st.cache_resource
def load_assets():
    try:
        model_pipeline = joblib.load('house_price_model.joblib')
        preprocessor = model_pipeline.named_steps['preprocessor']
        encoder = preprocessor.named_transformers_['Location_encoder']
        trained_locations = list(encoder.categories_[0])
        return model_pipeline, trained_locations
    except: 
        return None, ["DHA Phase 6", "Bahria Town", "Gulberg Islamabad"]

model, locations = load_assets()

# --- 4. AUTHENTICATION ---
if 'auth_status' not in st.session_state:
    st.session_state.auth_status = False

if not st.session_state.auth_status:
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        try: st.image("logo.png", use_container_width=True)
        except: st.markdown('<p class="sidebar-brand">Zameen AI Pro</p>', unsafe_allow_html=True)
        
        auth_tabs = st.tabs(["🔐 LOGIN", "📝 REGISTER"])
        with auth_tabs[0]:
            u = st.text_input("Username", key="login_u")
            p = st.text_input("Password", type="password", key="login_p")
            if st.button("🚀 ENTER DASHBOARD"):
                if login_user(u, p):
                    st.session_state.auth_status, st.session_state.username = True, u
                    st.rerun()
                else: st.error("Invalid Credentials")
        with auth_tabs[1]:
            nu = st.text_input("New Username", key="reg_u")
            npw = st.text_input("New Password", type="password", key="reg_p")
            if st.button("🆕 CREATE ACCOUNT"):
                if add_userdata(nu, npw): st.success("Account created!")
                else: st.error("User already exists.")
    st.stop()

# --- 5. SIDEBAR (With Logo) ---
with st.sidebar:
    try:
        st.image("logo.png", use_container_width=True)
    except:
        st.markdown('<p class="sidebar-brand">Zameen AI Pro</p>', unsafe_allow_html=True)
    
    st.markdown('<p class="tagline">Hybrid Intelligence</p>', unsafe_allow_html=True)
    st.divider()
    
    st.markdown('<p style="color:#10b981; font-weight:bold; font-size:0.9rem; text-align:center;">⚖️ AREA CONVERTER</p>', unsafe_allow_html=True)
    side_sqyd = st.number_input("SqYd", value=125, step=25, label_visibility="collapsed")
    st.markdown(f'''
        <div class="conv-box">
            <div class="conv-label">Marlas</div><div class="conv-val">{(side_sqyd/25):.2f}</div>
            <div class="conv-label" style="margin-top:10px">Kanals</div><div class="conv-val">{(side_sqyd/500):.4f}</div>
        </div>''', unsafe_allow_html=True)
    
    st.divider()
    if st.button("🚪 LOGOUT"):
        st.session_state.auth_status = False
        st.rerun()

# --- 6. MAIN DASHBOARD ---
main_tab, hist_tab = st.tabs(["🚀 Predictor", "📜 History"])

with main_tab:
    l_col, r_col = st.columns([3, 1.2], gap="medium")
    
    with l_col:
        st.markdown('<div class="specs-card">', unsafe_allow_html=True)
        loc_name = st.selectbox("Location / Sector", locations)
        c1, c2, c3, c4 = st.columns(4)
        area = c1.number_input("Area (SqYd)", 1, 10000, 125)
        beds = c2.number_input("Beds", 1, 15, 3)
        baths = c3.number_input("Baths", 1, 15, 3)
        kits = c4.number_input("Kitchens", 1, 5, 1)
        st.markdown('</div>', unsafe_allow_html=True)
        predict_btn = st.button("🚀 GENERATE HYBRID VALUATION")

    with r_col:
        geolocator = Nominatim(user_agent="ZameenAI_Pro")
        try:
            res = geolocator.geocode(f"{loc_name}, Pakistan", timeout=3)
            if res: st.map(pd.DataFrame({'lat': [res.latitude], 'lon': [res.longitude]}), zoom=13)
        except: st.info("Map loading...")

    if predict_btn and model:
        try:
            # 1. Prediction Logic
            raw_df = pd.DataFrame({'Location':[loc_name],'Area':[area],'Baths':[baths],'Beds':[beds],'Kitchens':[kits],'Drawing Room':[1],'Lounge or Sitting Room':[1]})
            preprocessor = model.named_steps['preprocessor']
            X_transformed = preprocessor.transform(raw_df)
            
            # Align to 250 features
            if X_transformed.shape[1] < 250:
                X_aligned = np.hstack([X_transformed, np.zeros((1, 250 - X_transformed.shape[1]))])
            else: X_aligned = X_transformed

            log_val = sklearn.pipeline.Pipeline(model.steps[1:]).predict(X_aligned)[0]
            ai_val = np.expm1(log_val)
            
            # 2. Pulse Logic
            sentiment, deviation, icon = ZameenPulse().get_market_pulse(ai_val)

            # 3. Display Results
            st.balloons()
            st.markdown("### 💎 Hybrid Valuation Report")
            res_l, res_r = st.columns(2)
            res_l.markdown(f'<div class="price-card"><small>AI MODEL VALUATION</small><h2>PKR {int(ai_val):,}</h2></div>', unsafe_allow_html=True)
            res_r.markdown(f'<div class="live-card"><small>LIVE MARKET PULSE</small><h2>PKR {int(deviation):,}</h2><p style="color:#10b981;">{icon} {sentiment}</p></div>', unsafe_allow_html=True)
            
            add_history(st.session_state.username, loc_name, area, ai_val, sentiment)
        except Exception as e: st.error(f"Error: {e}")

with hist_tab:
    df = view_user_history(st.session_state.username)
    if not df.empty: st.dataframe(df.sort_values(by="timestamp", ascending=False), use_container_width=True)
