import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn
import warnings
import json
import tempfile
import os
import random
import statistics
from streamlit_google_auth import Authenticate
from database_manager import *

# --- 1. SUPPRESS SKLEARN VERSION WARNINGS ---
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# --- 2. CORE COMPATIBILITY PATCHES ---
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list): pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

st.set_page_config(page_title="Zameen AI Pro | Hybrid Intelligence", layout="wide", page_icon="🏢")

try:
    init_db()
except Exception as e:
    st.error(f"Database Initialization Failed: {e}")

# --- 3. INITIALIZE SESSION STATE ---
if 'username' not in st.session_state: st.session_state.username = None
if 'connected' not in st.session_state: st.session_state.connected = False
if 'user_info' not in st.session_state: st.session_state.user_info = None

# --- 4. GOOGLE AUTH PERSISTENCE FIX ---
google_secrets = {
    "web": {
        "client_id": st.secrets["GOOGLE_CLIENT_ID"],
        "client_secret": st.secrets["GOOGLE_CLIENT_SECRET"],
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
}

# Check if the temp file still exists; if not, recreate it
if 'temp_credentials_path' not in st.session_state or not os.path.exists(st.session_state.temp_credentials_path):
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
        json.dump(google_secrets, temp_file)
        st.session_state.temp_credentials_path = temp_file.name

auth = Authenticate(
    secret_credentials_path=st.session_state.temp_credentials_path, 
    cookie_name='zameen_ai_pro_session',
    cookie_key=st.secrets["GOOGLE_CLIENT_SECRET"],
    redirect_uri="https://zameen-ai-pro.streamlit.app",
)

auth.check_authentification()

# --- 5. UI STYLING & ASSET LOADING ---
st.markdown("""
    <style>
    header {visibility: hidden;}
    .stApp { background-color: #020617; color: #ffffff; }
    [data-testid="stSidebar"] { background: #0f172a; border-right: 2px solid #10b981; }
    .sidebar-brand { font-size: 2.2rem !important; font-weight: 900 !important; background: linear-gradient(90deg, #10b981, #ffffff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; display: block; }
    .tagline { color: #10b981; font-size: 0.8rem; text-align: center; display: block; margin-top: -15px; margin-bottom: 20px; font-weight: bold; text-transform: uppercase; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    try:
        model = joblib.load('house_price_model.joblib')
        # Attempt to extract location categories from the OneHotEncoder step
        col_trans = next(step for step in model.named_steps.values() if isinstance(step, sklearn.compose.ColumnTransformer))
        encoder = next(trans[1] for trans in col_trans.transformers_ if 'OneHotEncoder' in str(type(trans[1])))
        return model, list(encoder.categories_[0])
    except:
        return None, ["DHA Phase 6", "Bahria Town", "Gulberg Islamabad", "E-11", "G-11"]

model, locations = load_assets()

# --- 6. AUTHENTICATION UI ---
if st.session_state.get('connected'):
    if st.session_state.get('user_info'):
        st.session_state.username = st.session_state['user_info'].get('email')
        add_google_userdata(st.session_state.username)

if not st.session_state.get('connected'):
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown('<div style="margin-top: 5rem;"><p class="sidebar-brand">Zameen AI Pro</p><p class="tagline">AI-Powered Property Valuation</p></div>', unsafe_allow_html=True)
        auth_tabs = st.tabs(["🌐 GOOGLE LOGIN", "📝 MANUAL ACCESS"])
        with auth_tabs[0]:
            auth.login()
            if st.session_state.get('connected'): st.rerun()
        with auth_tabs[1]:
            u = st.text_input("Username", key="login_u")
            p = st.text_input("Password", type="password", key="login_p")
            if st.button("🚀 LOGIN"):
                if login_user(u, p):
                    st.session_state.connected, st.session_state.username = True, u
                    st.rerun()
                else: st.error("Invalid Credentials")
    st.stop()

# --- 7. MAIN DASHBOARD ---
with st.sidebar:
    st.markdown('<p class="sidebar-brand">Zameen AI Pro</p>', unsafe_allow_html=True)
    if st.button("🚪 LOGOUT"):
        auth.logout()
        st.session_state.connected = False
        st.rerun()

main_tab, hist_tab = st.tabs(["🚀 Predictor", "📜 History"])

with main_tab:
    loc_name = st.selectbox("Location / Sector", locations)
    area_sqyd = st.number_input("Area (SqYd)", 1, 10000, 125)
    if st.button("🚀 GENERATE VALUATION"):
        # Dummy logic for example; replace with your prediction data dictionary
        st.success(f"Valuation generated for {loc_name}!")
        add_history(st.session_state.username, loc_name, area_sqyd, 15000000, "Stable")

with hist_tab:
    history_df = view_user_history(st.session_state.username)
    if not history_df.empty:
        # UPDATED: Use width='stretch' instead of use_container_width=True
        st.dataframe(history_df.sort_values(by="timestamp", ascending=False), width="stretch")
    else:
        st.info("No valuation history found.")
