import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn
import sklearn.compose._column_transformer
from geopy.geocoders import Nominatim
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

# --- 3. HARD-CODED REDIRECT LOGIC ---
# Using a more direct approach to stop 'localhost' from leaking into production
if st.get_option("browser.gatherUsageStats") == False: # A common trick to detect local runs
    REDIRECT_URI = "http://localhost:8501/callback"
else:
    REDIRECT_URI = "https://zameen-ai-pro.streamlit.app/callback"

st.set_page_config(page_title="Zameen AI Pro | Hybrid Intelligence", layout="wide", page_icon="🏢")

# Initialize Database
try:
    init_db()
except:
    pass

# --- 4. SESSION STATE & CALLBACK HANDLING ---
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

# --- 5. THE EMERALD UI CSS (REFINED WHITE TEXT) ---
st.markdown("""
    <style>
    header {visibility: hidden;}
    .stApp { background-color: #020617; color: #ffffff; }
    [data-testid="stSidebar"] { background: #0f172a; border-right: 2px solid #10b981; }
    .sidebar-brand { font-size: 2.2rem !important; font-weight: 900 !important; background: linear-gradient(90deg, #10b981, #ffffff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; display: block; }
    .tagline { color: #10b981; font-size: 0.8rem; text-align: center; display: block; margin-top: -15px; margin-bottom: 20px; font-weight: bold; text-transform: uppercase; }
    
    /* White text for all Tab labels */
    button[data-baseweb="tab"] p { color: white !important; font-size: 14px !important; font-weight: 600 !important; }
    button[data-baseweb="tab"][aria-selected="true"] { border-bottom: 3px solid #10b981 !important; }

    /* White text for all input labels */
    label[data-testid="stWidgetLabel"] p { color: white !important; font-weight: bold !important; }

    /* Login Button Styling */
    div.stButton > button { background-color: #0f172a !important; color: #10b981 !important; border: 2px solid #10b981 !important; border-radius: 8px; font-weight: 800 !important; width: 100% !important; padding: 10px !important; }
    div.stButton > button:hover { background-color: #10b981 !important; color: #020617 !important; box-shadow: 0 0 20px #10b981; }
    
    .specs-card { background-color: #0f172a; padding: 1.5rem !important; border-radius: 12px; border: 1px solid #10b981; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 6. AUTHENTICATION UI ---
if not st.session_state.auth_status:
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown('<div style="margin-top: 5rem;"><p class="sidebar-brand">Zameen AI Pro</p><p class="tagline">Hybrid Intelligence Valuation</p></div>', unsafe_allow_html=True)
        auth_tabs = st.tabs(["🌐 GOOGLE ACCESS", "🔐 MANUAL LOGIN", "📝 REGISTER"])
        
        with auth_tabs[0]:
            try:
                auth_url = workos_client.user_management.get_authorization_url(
                    redirect_uri=REDIRECT_URI,
                    provider="google"
                )
                # target="_top" is necessary to avoid the iframe error in your previous screenshots
                st.markdown(
                    f"""
                    <a href="{auth_url}" target="_top" style="text-decoration: none;">
                        <div style="background-color: #10b981; color: #020617; padding: 18px; text-align: center; border-radius: 8px; font-weight: 900; cursor: pointer; margin-top: 10px;">
                            CONTINUE WITH GOOGLE
                        </div>
                    </a>
                    """, 
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Auth URL Error: {e}")

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
# ... Predictor logic remains the same ...
