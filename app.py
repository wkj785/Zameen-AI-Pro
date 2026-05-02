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
from workos import WorkOSClient
from database_manager import *

# --- 1. CORE COMPATIBILITY PATCHES ---
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list): pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

# --- 2. WORKOS SETUP ---
workos_client = WorkOSClient(
    api_key=st.secrets["WORKOS_API_KEY"],
    client_id=st.secrets["WORKOS_CLIENT_ID"]
)

# --- FIXED REDIRECT LOGIC ---
# This ensures that on the web, it ALWAYS uses the streamlit.app URL
if "streamlit.app" in st.get_option("browser.serverAddress") or "uhcnnc2afavzqsnsjtwgjg" in st.get_option("browser.serverAddress"):
    REDIRECT_URI = "https://zameen-ai-pro.streamlit.app/callback"
else:
    REDIRECT_URI = "http://localhost:8501/callback"

st.set_page_config(page_title="Zameen AI Pro | Hybrid Intelligence", layout="wide", page_icon="🏢")

# Initialize Database
try:
    init_db()
except:
    pass

# --- 3. SESSION STATE & CALLBACK HANDLING ---
if 'auth_status' not in st.session_state:
    st.session_state.auth_status = False

query_params = st.query_params
if "code" in query_params and not st.session_state.auth_status:
    try:
        response = workos_client.user_management.authenticate_with_code(
            client_id=st.secrets["WORKOS_CLIENT_ID"],
            code=query_params["code"],
        )
        st.session_state.username = response.user.email
        st.session_state.auth_status = True
        add_google_userdata(response.user.email) 
        st.query_params.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Authentication Failed: {e}")

# --- 4. THE EMERALD UI CSS ---
st.markdown("""
    <style>
    header {visibility: hidden;}
    .stApp { background-color: #020617; color: #ffffff; }
    [data-testid="stSidebar"] { background: #0f172a; border-right: 2px solid #10b981; }
    .sidebar-brand { font-size: 2.2rem !important; font-weight: 900 !important; background: linear-gradient(90deg, #10b981, #ffffff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; display: block; }
    .tagline { color: #10b981; font-size: 0.8rem; text-align: center; display: block; margin-top: -15px; margin-bottom: 20px; font-weight: bold; text-transform: uppercase; }
    
    div.stButton > button, .workos-btn { background-color: #0f172a !important; color: #10b981 !important; border: 2px solid #10b981 !important; border-radius: 8px; font-weight: 800 !important; width: 100% !important; padding: 18px !important; font-size: 1.1rem !important; text-align: center; text-decoration: none; display: block; }
    div.stButton > button:hover, .workos-btn:hover { background-color: #10b981 !important; color: #020617 !important; box-shadow: 0 0 20px #10b981; transition: 0.3s; }
    
    .specs-card { background-color: #0f172a; padding: 1.5rem !important; border-radius: 12px; border: 1px solid #10b981; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 5. ASSET LOADING ---
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

# --- 6. AUTHENTICATION UI ---
if not st.session_state.auth_status:
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown('<div style="margin-top: 5rem;"><p class="sidebar-brand">Zameen AI Pro</p><p class="tagline">Hybrid Intelligence Valuation</p></div>', unsafe_allow_html=True)
        auth_tabs = st.tabs(["🌐 GOOGLE ACCESS", "🔐 MANUAL LOGIN", "📝 REGISTER"])
        
        with auth_tabs[0]:
            auth_url = workos_client.user_management.get_authorization_url(
                redirect_uri=REDIRECT_URI,
                provider="google"
            )
            # IMPORTANT: Added target="_top" to ensure it breaks out of the Streamlit frame
            st.markdown(f'<a href="{auth_url}" target="_top" class="workos-btn">CONTINUE WITH GOOGLE</a>', unsafe_allow_html=True)

        with auth_tabs[1]:
            u = st.text_input("Username", key="login_u")
            p = st.text_input("Password", type="password", key="login_p")
            if st.button("🚀 ENTER DASHBOARD"):
                if login_user(u, p):
                    st.session_state.auth_status, st.session_state.username = True, u
                    st.rerun()
                else: st.error("Invalid Credentials")
        
        with auth_tabs[2]:
            nu = st.text_input("New Username", key="reg_u")
            npw = st.text_input("New Password", type="password", key="reg_p")
            if st.button("🆕 CREATE ACCOUNT"):
                if add_userdata(nu, npw): st.success("Account created! Please login.")
                else: st.error("User already exists.")
    st.stop()

# --- 7. DASHBOARD CONTENT ---
with st.sidebar:
    st.markdown('<p class="sidebar-brand">Zameen AI Pro</p>', unsafe_allow_html=True)
    st.write(f"Logged in: **{st.session_state.username}**")
    if st.button("🚪 LOGOUT"):
        st.session_state.auth_status = False
        st.rerun()

main_tab, hist_tab = st.tabs(["🚀 Predictor", "📜 History"])
# ... [Insert your Predictor/History logic here from previous versions]
