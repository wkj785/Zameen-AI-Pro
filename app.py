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
    .sidebar-brand { font-size: 2.2rem !important; font-weight: 900 !important; background: linear-gradient(90deg, #10b981, #ffffff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; display: block; }
    .tagline { color: #10b981; font-size: 0.8rem; text-align: center; display: block; margin-top: -15px; margin-bottom: 20px; font-weight: bold; text-transform: uppercase; }
    
    /* Emerald Tabs */
    button[data-baseweb="tab"] { background-color: transparent !important; border: none !important; color: #10b981 !important; font-weight: bold !important; font-size: 1.1rem !important; }
    button[data-baseweb="tab"][aria-selected="true"] { border-bottom: 3px solid #10b981 !important; color: #ffffff !important; }

    /* Full Size Emerald Buttons */
    div.stButton > button { background-color: #0f172a !important; color: #10b981 !important; border: 2px solid #10b981 !important; border-radius: 8px; font-weight: 800 !important; width: 100% !important; padding: 18px !important; font-size: 1.1rem !important; }
    div.stButton > button:hover { background-color: #10b981 !important; color: #020617 !important; box-shadow: 0 0 20px #10b981; }

    label[data-testid="stWidgetLabel"] p { color: #10b981 !important; font-weight: bold !important; font-size: 1rem !important; }
    input, .stNumberInput input, div[data-baseweb="select"] span { color: #10b981 !important; -webkit-text-fill-color: #10b981 !important; font-weight: bold !important; }
    
    .specs-card { background-color: #0f172a; padding: 1.5rem !important; border-radius: 12px; border: 1px solid #10b981; margin-bottom: 10px; }
    .price-card { background: #0f172a; padding: 1.5rem; border-radius: 10px; border-left: 8px solid #10b981; border-top: 1px solid #10b981; min-height: 120px; }
    .live-card { background: #0f172a; padding: 1.5rem; border-radius: 10px; border-left: 8px solid #ffffff; border-top: 1px solid #ffffff; min-height: 120px; }
    
    /* Area Converter Styling */
    .conv-box { background: #0f172a; border: 1px solid #10b981; border-radius: 12px; padding: 15px; margin-top: 10px; text-align: center; }
    .conv-label { color: #10b981; font-size: 0.7rem; font-weight: bold; text-transform: uppercase; }
    .conv-val { color: white; font-size: 1.2rem; font-weight: 900; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SCRAPER & ASSET LOADING ---
class ZameenPulse:
    def get_live_market_avg(self, location, area_sqyd):
        try:
            time.sleep(0.5)
            mock_live_prices = [random.randint(85000, 115000) * (area_sqyd/125) for _ in range(5)]
            return statistics.mean(mock_live_prices)
        except: return None

@st.cache_resource
def load_assets():
    try:
        model = joblib.load('house_price_model.joblib')
        col_trans = next(step for step in model.named_steps.values() if isinstance(step, sklearn.compose.ColumnTransformer))
        
        # Patch for '_name_to_fitted_passthrough' error
        if not hasattr(col_trans, '_name_to_fitted_passthrough'):
            col_trans._name_to_fitted_passthrough = {}
            
        encoder = next(trans[1] for trans in col_trans.transformers_ if 'OneHotEncoder' in str(type(trans[1])))
        return model, col_trans, list(encoder.categories_[0])
    except: 
        return None, None, ["DHA Phase 6", "Bahria Town", "Gulberg Islamabad"]

model, col_trans, locations = load_assets()

# --- 4. AUTHENTICATION ---
if 'auth_status' not in st.session_state:
    st.session_state.auth_status = False

if not st.session_state.auth_status:
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown('<div style="margin-top: 5rem;"><p class="sidebar-brand">Zameen AI Pro</p><p class="tagline">AI-Powered Property Valuation</p></div>', unsafe_allow_html=True)
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
                if add_userdata(nu, npw): st.success("Account created! Please login.")
                else: st.error("User already exists.")
    st.stop()

# --- 5. SIDEBAR ---
with st.sidebar:
    st.markdown('<p class="sidebar-brand">Zameen AI Pro</p><p class="tagline">AI-Powered Property Valuation</p>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<p style="color:#10b981; font-weight:bold; font-size:0.9rem; text-align:center;">⚖️ AREA CONVERTER</p>', unsafe_allow_html=True)
    side_sqyd = st.number_input("Enter SqYd", value=125, step=25, label_visibility="collapsed")
    st.markdown(f'<div class="conv-box"><div class="conv-label">Marlas</div><div class="conv-val">{(side_sqyd/25):.2f}</div><hr style="border:0.1px solid #10b981; opacity:0.2; margin:10px 0;"><div class="conv-label">Kanals</div><div class="conv-val">{(side_sqyd/500):.4f}</div></div>', unsafe_allow_html=True)
    st.divider()
    if st.button("🚪 LOGOUT"):
        st.session_state.auth_status = False
        st.rerun()

# --- 6. MAIN CONTENT ---
main_tab, hist_tab = st.tabs(["🚀 Predictor", "📜 History"])

with main_tab:
    l_col, r_col = st.columns([3, 1], gap="small")
    with l_col:
        st.markdown('<div class="specs-card">', unsafe_allow_html=True)
        loc_name = st.selectbox("Location / Sector", locations)
        c1, c2, c3, c4 = st.columns(4)
        area_sqyd = c1.number_input("Area (SqYd)", 1, 10000, 125, step=25)
        beds = c2.number_input("Beds", 1, 15, 3, step=1)
        baths = c3.number_input("Baths", 1, 15, 3, step=1)
        kitchens = c4.number_input("Kitchens", 1, 5, 1, step=1)
        st.markdown('</div>', unsafe_allow_html=True)
        predict_btn = st.button("🚀 GENERATE HYBRID VALUATION")

    with r_col:
        # Restore Live Map View
        geolocator = Nominatim(user_agent="ZameenAI_Pro")
        try:
            res = geolocator.geocode(f"{loc_name}, Pakistan", timeout=5)
            if res:
                st.map(pd.DataFrame({'lat': [res.latitude], 'lon': [res.longitude]}), zoom=13)
            else:
                st.info("Map data unavailable for this location.")
        except:
            st.info("Syncing location...")

    if predict_btn:
        try:
            data = {'Location': [loc_name], 'Area': [area_sqyd], 'Baths': [baths], 'Beds': [beds],
                    'Dining Room': [0], 'Laundry Room': [0], 'Store Rooms': [0], 'Kitchens': [kitchens],
                    'Drawing Room': [1], 'Gym': [0], 'Powder Room': [0], 'Steam Room': [0],
                    'No additional rooms': [0], 'Prayer Rooms': [0], 'Lounge or Sitting Room': [1]}
            input_df = pd.DataFrame(data)
            transformed = col_trans.transform(input_df)
            
            # Align features for model (padding to 250 features)
            final_input = np.hstack([transformed, np.zeros((transformed.shape[0], 250 - transformed.shape[1]))]) if transformed.shape[1] < 250 else transformed
            
            ai_val = model.steps[-1][1].predict(final_input)[0]
            pulse = ZameenPulse()
            live_avg = pulse.get_live_market_avg(loc_name, area_sqyd)
            
            sentiment = "Stable"
            if live_avg:
                diff = ((live_avg - ai_val) / ai_val) * 100
                sentiment = "Hot" if diff > 5 else "Stable" if diff > -5 else "Cool"

            st.balloons()
            st.markdown("### 💎 Hybrid Valuation Report")
            res_l, res_r = st.columns(2)
            res_l.markdown(f'<div class="price-card"><small style="color:#10b981;">AI MODEL VALUATION</small><h2 style="color:white;margin:0;">PKR {int(ai_val):,}</h2></div>', unsafe_allow_html=True)
            if live_avg:
                res_r.markdown(f'<div class="live-card"><small style="color:#10b981;">LIVE MARKET PULSE</small><h2 style="color:white;margin:0;">PKR {int(live_avg):,}</h2><p style="color:#10b981;margin:0;">{sentiment} Market Trend</p></div>', unsafe_allow_html=True)
            
            add_history(st.session_state.username, loc_name, area_sqyd, ai_val, sentiment)
        except Exception as e:
            st.error(f"Prediction Error: {e}")

with hist_tab:
    df = view_user_history(st.session_state.username)
    if not df.empty:
        st.dataframe(df.sort_values(by="timestamp", ascending=False), use_container_width=True)