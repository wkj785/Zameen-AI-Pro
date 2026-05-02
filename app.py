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
from streamlit_google_auth import Authenticate
from database_manager import *

# --- 1. CORE COMPATIBILITY PATCHES ---
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list): pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

st.set_page_config(page_title="Zameen AI Pro | Hybrid Intelligence", layout="wide", page_icon="🏢")
init_db()

# --- 2. INITIALIZE SESSION STATE ---
# Fixes AttributeError by ensuring these exist before the script runs logic
if 'username' not in st.session_state:
    st.session_state.username = None
if 'connected' not in st.session_state:
    st.session_state.connected = False

# --- 3. THE ULTIMATE EMERALD UI CSS ---
st.markdown("""
    <style>
    header {visibility: hidden;}
    .stApp { background-color: #020617; color: #ffffff; }
    [data-testid="stSidebar"] { background: #0f172a; border-right: 2px solid #10b981; }
    .sidebar-brand { font-size: 2.2rem !important; font-weight: 900 !important; background: linear-gradient(90deg, #10b981, #ffffff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; display: block; }
    .tagline { color: #10b981; font-size: 0.8rem; text-align: center; display: block; margin-top: -15px; margin-bottom: 20px; font-weight: bold; text-transform: uppercase; }
    
    button[data-baseweb="tab"] { background-color: transparent !important; color: #10b981 !important; font-weight: bold !important; }
    button[data-baseweb="tab"][aria-selected="true"] { border-bottom: 3px solid #10b981 !important; color: #ffffff !important; }

    div.stButton > button { background-color: #0f172a !important; color: #10b981 !important; border: 2px solid #10b981 !important; border-radius: 8px; width: 100% !important; padding: 10px !important; }
    div.stButton > button:hover { background-color: #10b981 !important; color: #020617 !important; box-shadow: 0 0 20px #10b981; }

    .specs-card { background-color: #0f172a; padding: 1.5rem; border-radius: 12px; border: 1px solid #10b981; margin-bottom: 10px; }
    .price-card { background: #0f172a; padding: 1.5rem; border-radius: 10px; border-left: 8px solid #10b981; border-top: 1px solid #10b981; }
    .live-card { background: #0f172a; padding: 1.5rem; border-radius: 10px; border-left: 8px solid #ffffff; border-top: 1px solid #ffffff; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. ASSET LOADING & PULSE ---
class ZameenPulse:
    def get_live_market_avg(self, location, area_sqyd):
        try:
            mock_live_prices = [random.randint(85000, 115000) * (area_sqyd/125) for _ in range(5)]
            return statistics.mean(mock_live_prices)
        except: return None

@st.cache_resource
def load_assets():
    try:
        model = joblib.load('house_price_model.joblib')
        col_trans = next(step for step in model.named_steps.values() if isinstance(step, sklearn.compose.ColumnTransformer))
        if not hasattr(col_trans, '_name_to_fitted_passthrough'):
            col_trans._name_to_fitted_passthrough = {}
        encoder = next(trans[1] for trans in col_trans.transformers_ if 'OneHotEncoder' in str(type(trans[1])))
        return model, col_trans, list(encoder.categories_[0])
    except: 
        return None, None, ["DHA Phase 6", "Bahria Town", "Gulberg Islamabad"]

model, col_trans, locations = load_assets()

import json
import tempfile
import os

# --- 5. HYBRID AUTHENTICATION (GOOGLE + MANUAL) ---

# 1. Create the credentials dictionary
google_secrets = {
    "web": {
        "client_id": st.secrets["GOOGLE_CLIENT_ID"],
        "client_secret": st.secrets["GOOGLE_CLIENT_SECRET"],
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
}

# 2. Write the dictionary to a temporary JSON file so the library can read a "Path"
with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
    json.dump(google_secrets, temp_file)
    temp_credentials_path = temp_file.name

# 3. Initialize Authenticate using the path to that temporary file
auth = Authenticate(
    secret_credentials_path=temp_credentials_path, # Now it's a str (Path), not a dict
    cookie_name='zameen_ai_pro_session',
    cookie_key=st.secrets["GOOGLE_CLIENT_SECRET"],
    redirect_uri="https://zameen-ai-pro.streamlit.app",
)

auth.check_authentification()

# 4. Clean up the temporary file (optional but good practice)
# os.unlink(temp_credentials_path)

# --- 6. SIDEBAR ---
with st.sidebar:
    st.markdown('<p class="sidebar-brand">Zameen AI Pro</p>', unsafe_allow_html=True)
    
    if st.session_state.username:
        if st.session_state.get('user_info'):
            st.image(st.session_state['user_info'].get('picture'), width=70)
            st.write(f"Welcome, {st.session_state['user_info'].get('name')}")
        else:
            st.write(f"Logged in as: {st.session_state.username}")
    
    st.divider()
    if st.button("🚪 LOGOUT"):
        auth.logout()
        st.session_state.connected = False
        st.session_state.username = None
        st.rerun()

# --- 7. MAIN CONTENT ---
main_tab, hist_tab = st.tabs(["🚀 Predictor", "📜 History"])

with main_tab:
    st.markdown('<div class="specs-card">', unsafe_allow_html=True)
    loc_name = st.selectbox("Location / Sector", locations)
    c1, c2, c3, c4 = st.columns(4)
    area_sqyd = c1.number_input("Area (SqYd)", 1, 10000, 125)
    beds = c2.number_input("Beds", 1, 15, 3)
    baths = c3.number_input("Baths", 1, 15, 3)
    kitchens = c4.number_input("Kitchens", 1, 5, 1)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("🚀 GENERATE HYBRID VALUATION"):
        try:
            # Model data preparation
            data = {'Location': [loc_name], 'Area': [area_sqyd], 'Baths': [baths], 'Beds': [beds],
                    'Dining Room': [0], 'Laundry Room': [0], 'Store Rooms': [0], 'Kitchens': [kitchens],
                    'Drawing Room': [1], 'Gym': [0], 'Powder Room': [0], 'Steam Room': [0],
                    'No additional rooms': [0], 'Prayer Rooms': [0], 'Lounge or Sitting Room': [1]}
            input_df = pd.DataFrame(data)
            transformed = col_trans.transform(input_df)
            final_input = np.hstack([transformed, np.zeros((1, 250 - transformed.shape[1]))])
            
            ai_val = model.steps[-1][1].predict(final_input)[0]
            live_avg = ZameenPulse().get_live_market_avg(loc_name, area_sqyd)
            
            st.balloons()
            st.markdown("### 💎 Hybrid Valuation Report")
            res_l, res_r = st.columns(2)
            res_l.markdown(f'<div class="price-card"><small>AI VALUATION</small><h2>PKR {int(ai_val):,}</h2></div>', unsafe_allow_html=True)
            if live_avg:
                sentiment = "Hot" if live_avg > ai_val else "Stable"
                res_r.markdown(f'<div class="live-card"><small>LIVE PULSE</small><h2>PKR {int(live_avg):,}</h2><p>{sentiment} Market</p></div>', unsafe_allow_html=True)
            
            add_history(st.session_state.username, loc_name, area_sqyd, ai_val, sentiment if live_avg else "Stable")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

with hist_tab:
    if st.session_state.username:
        df = view_user_history(st.session_state.username)
        if not df.empty:
            st.dataframe(df.sort_values(by="timestamp", ascending=False), use_container_width=True)
        else:
            st.info("No valuation history found yet.")
