import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn
import sklearn.compose._column_transformer
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

# --- 3. PRODUCTION REDIRECT CHECK ---
# Using the gatherUsageStats flag is the most reliable way to detect Streamlit Cloud
if st.get_option("browser.gatherUsageStats") == False:
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

# --- 5. THE EMERALD UI CSS (COMPLETE WHITE TEXT OVERRIDE) ---
st.markdown("""
    <style>
    header {visibility: hidden;}
    .stApp { background-color: #020617; color: #ffffff; }
    [data-testid="stSidebar"] { background: #0f172a; border-right: 2px solid #10b981; }
    
    /* Brand Styling */
    .sidebar-brand { font-size: 2.2rem !important; font-weight: 900 !important; background: linear-gradient(90deg, #10b981, #ffffff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; display: block; }
    .tagline { color: #10b981; font-size: 0.8rem; text-align: center; display: block; margin-top: -15px; margin-bottom: 20px; font-weight: bold; text-transform: uppercase; }
    
    /* White text for all Tab labels (Auth & Dashboard) */
    button[data-baseweb="tab"] p { color: white !important; font-size: 14px !important; font-weight: 600 !important; }
    button[data-baseweb="tab"][aria-selected="true"] { border-bottom: 3px solid #10b981 !important; }

    /* White text for all input labels and sidebar text */
    label[data-testid="stWidgetLabel"] p { color: white !important; font-weight: bold !important; }
    [data-testid="stSidebar"] p { color: white !important; }

    /* Custom Button Styling (Main Buttons) */
    div.stButton > button { background-color: #0f172a !important; color: #10b981 !important; border: 2px solid #10b981 !important; border-radius: 8px; font-weight: 800 !important; width: 100% !important; padding: 10px !important; }
    div.stButton > button:hover { background-color: #10b981 !important; color: #020617 !important; box-shadow: 0 0 20px #10b981; transition: 0.3s; }
    
    /* Secondary Containers */
    .specs-card { background-color: #0f172a; padding: 1.5rem !important; border-radius: 12px; border: 1px solid #10b981; margin-bottom: 10px; }
    .price-card { background: #0f172a; padding: 1.5rem; border-radius: 10px; border-left: 8px solid #10b981; border-top: 1px solid #10b981; }
    </style>
    """, unsafe_allow_html=True)

# --- 6. ASSET LOADING ---
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

# --- 7. AUTHENTICATION UI ---
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

# --- 8. DASHBOARD CONTENT ---
with st.sidebar:
    st.markdown('<p class="sidebar-brand">Zameen AI Pro</p>', unsafe_allow_html=True)
    st.write(f"User: **{st.session_state.username}**")
    st.divider()
    if st.button("🚪 LOGOUT SESSION"):
        st.session_state.auth_status = False
        st.session_state.username = None
        st.rerun()

main_tab, hist_tab = st.tabs(["🚀 PREDICTOR", "📜 VALUATION HISTORY"])

with main_tab:
    l_col, r_col = st.columns([3, 1], gap="small")
    with l_col:
        st.markdown('<div class="specs-card">', unsafe_allow_html=True)
        loc_name = st.selectbox("Property Location", locations)
        c1, c2, c3, c4 = st.columns(4)
        area_sqyd = c1.number_input("Area (SqYd)", 1, 10000, 125, step=25)
        beds = c2.number_input("Beds", 1, 15, 3, step=1)
        baths = c3.number_input("Baths", 1, 15, 3, step=1)
        kitchens = c4.number_input("Kitchens", 1, 5, 1, step=1)
        st.markdown('</div>', unsafe_allow_html=True)
        predict_btn = st.button("🚀 CALCULATE PRICE")

    if predict_btn:
        try:
            # Basic calculation placeholder
            ai_val = (area_sqyd * 15500) + (beds * 550000) + (baths * 250000)
            st.balloons()
            st.markdown(f'<div class="price-card"><small style="color:#10b981;">HYBRID ESTIMATE</small><h2 style="color:white;margin:0;">PKR {int(ai_val):,}</h2></div>', unsafe_allow_html=True)
            add_history(st.session_state.username, loc_name, area_sqyd, ai_val, "Stable")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

with hist_tab:
    df = view_user_history(st.session_state.username)
    if not df.empty:
        st.dataframe(df.sort_values(by="timestamp", ascending=False), use_container_width=True)
    else:
        st.info("No records found in your valuation history.")
