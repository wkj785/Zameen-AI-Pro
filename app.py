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

# MATCHING SCREENSHOT (25).png: Adding /callback to the URIs
if not st.get_option("browser.serverAddress") or "localhost" in st.get_option("browser.serverAddress"):
    REDIRECT_URI = "http://localhost:8501/callback"
else:
    REDIRECT_URI = "https://zameen-ai-pro.streamlit.app/callback"

st.set_page_config(page_title="Zameen AI Pro | Hybrid Intelligence", layout="wide", page_icon="🏢")

# Initialize Database
try:
    init_db()
except:
    pass

# --- 3. SESSION STATE & CALLBACK HANDLING ---
if 'auth_status' not in st.session_state:
    st.session_state.auth_status = False

# Listener for the 'code' returned to the /callback path
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
    
    button[data-baseweb="tab"] { background-color: transparent !important; border: none !important; color: #10b981 !important; font-weight: bold !important; font-size: 1.1rem !important; }
    button[data-baseweb="tab"][aria-selected="true"] { border-bottom: 3px solid #10b981 !important; color: #ffffff !important; }

    div.stButton > button, .workos-btn { background-color: #0f172a !important; color: #10b981 !important; border: 2px solid #10b981 !important; border-radius: 8px; font-weight: 800 !important; width: 100% !important; padding: 18px !important; font-size: 1.1rem !important; text-align: center; text-decoration: none; display: block; }
    div.stButton > button:hover, .workos-btn:hover { background-color: #10b981 !important; color: #020617 !important; box-shadow: 0 0 20px #10b981; transition: 0.3s; }

    label[data-testid="stWidgetLabel"] p { color: #10b981 !important; font-weight: bold !important; font-size: 1rem !important; }
    input, .stNumberInput input, div[data-baseweb="select"] span { color: #10b981 !important; font-weight: bold !important; }
    
    .specs-card { background-color: #0f172a; padding: 1.5rem !important; border-radius: 12px; border: 1px solid #10b981; margin-bottom: 10px; }
    .price-card { background: #0f172a; padding: 1.5rem; border-radius: 10px; border-left: 8px solid #10b981; border-top: 1px solid #10b981; }
    .live-card { background: #0f172a; padding: 1.5rem; border-radius: 10px; border-left: 8px solid #ffffff; border-top: 1px solid #ffffff; }
    
    .conv-box { background: #0f172a; border: 1px solid #10b981; border-radius: 12px; padding: 15px; margin-top: 10px; text-align: center; }
    .conv-label { color: #10b981; font-size: 0.7rem; font-weight: bold; text-transform: uppercase; }
    .conv-val { color: white; font-size: 1.2rem; font-weight: 900; }
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
            st.markdown(f'<a href="{auth_url}" target="_self" class="workos-btn">CONTINUE WITH GOOGLE</a>', unsafe_allow_html=True)

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

# --- 7. DASHBOARD SIDEBAR ---
with st.sidebar:
    st.markdown('<p class="sidebar-brand">Zameen AI Pro</p><p class="tagline">AI-Powered Valuation</p>', unsafe_allow_html=True)
    st.write(f"Logged in: **{st.session_state.username}**")
    st.divider()
    st.markdown('<p style="color:#10b981; font-weight:bold; font-size:0.9rem; text-align:center;">⚖️ AREA CONVERTER</p>', unsafe_allow_html=True)
    side_sqyd = st.number_input("Enter SqYd", value=125, step=25, label_visibility="collapsed")
    st.markdown(f'<div class="conv-box"><div class="conv-label">Marlas</div><div class="conv-val">{(side_sqyd/25):.2f}</div><hr style="border:0.1px solid #10b981; opacity:0.2; margin:10px 0;"><div class="conv-label">Kanals</div><div class="conv-val">{(side_sqyd/500):.4f}</div></div>', unsafe_allow_html=True)
    st.divider()
    if st.button("🚪 LOGOUT"):
        st.session_state.auth_status = False
        st.session_state.username = None
        st.rerun()

# --- 8. MAIN CONTENT ---
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
        geolocator = Nominatim(user_agent="ZameenAI_Pro_v2")
        try:
            res = geolocator.geocode(f"{loc_name}, Pakistan", timeout=5)
            if res:
                st.map(pd.DataFrame({'lat': [res.latitude], 'lon': [res.longitude]}), zoom=13)
        except: st.info("Map data syncing...")

    if predict_btn:
        try:
            data = {'Location': [loc_name], 'Area': [area_sqyd], 'Baths': [baths], 'Beds': [beds],
                    'Dining Room': [0], 'Laundry Room': [0], 'Store Rooms': [0], 'Kitchens': [kitchens],
                    'Drawing Room': [1], 'Gym': [0], 'Powder Room': [0], 'Steam Room': [0],
                    'No additional rooms': [0], 'Prayer Rooms': [0], 'Lounge or Sitting Room': [1]}
            input_df = pd.DataFrame(data)
            transformed = col_trans.transform(input_df)
            final_input = np.hstack([transformed, np.zeros((transformed.shape[0], 250 - transformed.shape[1]))]) if transformed.shape[1] < 250 else transformed
            
            ai_val = model.steps[-1][1].predict(final_input)[0]
            st.balloons()
            st.markdown(f'<div class="price-card"><small style="color:#10b981;">AI MODEL VALUATION</small><h2 style="color:white;margin:0;">PKR {int(ai_val):,}</h2></div>', unsafe_allow_html=True)
            add_history(st.session_state.username, loc_name, area_sqyd, ai_val, "Stable")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

with hist_tab:
    df = view_user_history(st.session_state.username)
    if not df.empty:
        st.dataframe(df.sort_values(by="timestamp", ascending=False), width="stretch")
