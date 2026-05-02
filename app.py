import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn
import warnings
import json
import tempfile
import os
from streamlit_google_auth import Authenticate
from database_manager import *

# --- 1. SUPPRESS SKLEARN VERSION WARNINGS ---
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# --- 2. CORE COMPATIBILITY PATCHES ---
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list): pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

st.set_page_config(page_title="Zameen AI Pro | Hybrid Intelligence", layout="wide", page_icon="🏢")

# Database Init
try:
    init_db()
except Exception as e:
    st.error(f"Database Initialization Failed: {e}")

# --- 3. SESSION STATE ---
if 'username' not in st.session_state: st.session_state.username = None
if 'connected' not in st.session_state: st.session_state.connected = False

# --- 4. GOOGLE AUTH PERSISTENCE FIX ---
# Define secrets outside the logic
google_secrets = {
    "web": {
        "client_id": st.secrets["GOOGLE_CLIENT_ID"],
        "client_secret": st.secrets["GOOGLE_CLIENT_SECRET"],
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
}

# Ensure the temp file exists every time the script runs
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

# --- 5. STYLING ---
st.markdown("""
    <style>
    header {visibility: hidden;}
    .stApp { background-color: #020617; color: #ffffff; }
    [data-testid="stSidebar"] { background: #0f172a; border-right: 2px solid #10b981; }
    .sidebar-brand { font-size: 2.2rem !important; font-weight: 900 !important; background: linear-gradient(90deg, #10b981, #ffffff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; display: block; }
    </style>
    """, unsafe_allow_html=True)

# --- 6. MAIN LOGIC ---
if st.session_state.get('connected'):
    if st.session_state.get('user_info'):
        st.session_state.username = st.session_state['user_info'].get('email')
        add_google_userdata(st.session_state.username)

if not st.session_state.get('connected'):
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown('<div style="margin-top: 5rem;"><p class="sidebar-brand">Zameen AI Pro</p></div>', unsafe_allow_html=True)
        auth_tabs = st.tabs(["🌐 GOOGLE LOGIN", "📝 MANUAL ACCESS"])
        with auth_tabs[0]:
            auth.login()
        with auth_tabs[1]:
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.button("🚀 LOGIN"):
                if login_user(u, p):
                    st.session_state.connected, st.session_state.username = True, u
                    st.rerun()
    st.stop()

# --- 7. DASHBOARD ---
with st.sidebar:
    st.markdown('<p class="sidebar-brand">Zameen AI Pro</p>', unsafe_allow_html=True)
    if st.button("🚪 LOGOUT"):
        auth.logout()
        st.session_state.connected = False
        st.rerun()

main_tab, hist_tab = st.tabs(["🚀 Predictor", "📜 History"])

with hist_tab:
    history_df = view_user_history(st.session_state.username)
    if not history_df.empty:
        # Use width='stretch' to satisfy the new Streamlit requirements
        st.dataframe(history_df.sort_values(by="timestamp", ascending=False), width="stretch")
    else:
        st.info("No valuation history found.")
