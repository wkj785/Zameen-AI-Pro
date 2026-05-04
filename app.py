import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn
import sklearn.compose._column_transformer
from geopy.geocoders import Nominatim
import time
import random
from database_manager import * 

# --- 1. VERSION COMPATIBILITY PATCH ---
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list): pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

st.set_page_config(page_title="Zameen AI Pro", layout="wide", page_icon="🏢")
init_db()

# --- 2. EMERALD UI CSS (MATCHING SCREENSHOT 45) ---
st.markdown("""
    <style>
    header {visibility: hidden;}
    .stApp { background-color: #020617; color: #ffffff; }
    [data-testid="stSidebar"] { background: #0f172a; border-right: 2px solid #10b981; }
    .sidebar-brand { font-size: 2.2rem; font-weight: 900; background: linear-gradient(90deg, #10b981, #ffffff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; display: block; }
    div.stButton > button { background-color: #0f172a !important; color: #10b981 !important; border: 2px solid #10b981 !important; border-radius: 8px; width: 100%; padding: 15px; font-weight: bold; }
    div.stButton > button:hover { background-color: #10b981 !important; color: #020617 !important; }
    .price-card { background: #0f172a; padding: 1.5rem; border-radius: 10px; border-left: 8px solid #10b981; border-top: 1px solid #10b981; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. ASSET LOADING ---
@st.cache_resource
def load_assets():
    try:
        model_pipeline = joblib.load('house_price_model.joblib')
        preprocessor = model_pipeline.named_steps['preprocessor']
        encoder = preprocessor.named_transformers_['Location_encoder']
        if not hasattr(preprocessor, '_name_to_fitted_passthrough'):
            preprocessor._name_to_fitted_passthrough = {}
        return model_pipeline, list(encoder.categories_[0])
    except Exception as e: 
        return None, ["Karachi", "Lahore", "Islamabad"]

model, locations = load_assets()

# --- 4. AUTHENTICATION LOGIC ---
if 'auth_status' not in st.session_state:
    st.session_state.auth_status = False

if not st.session_state.auth_status:
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown('<p class="sidebar-brand">Zameen AI Pro</p>', unsafe_allow_html=True)
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("🚀 LOGIN"):
            if login_user(u, p):
                st.session_state.auth_status, st.session_state.username = True, u
                st.rerun()
            else: st.error("Invalid Credentials")
    st.stop()

# --- 5. SIDEBAR (MATCHING SCREENSHOT 45) ---
with st.sidebar:
    st.markdown('<p class="sidebar-brand">Zameen AI Pro</p>', unsafe_allow_html=True)
    st.divider()
    st.write("⚖️ AREA CONVERTER")
    side_sqyd = st.number_input("Enter SqYd", value=125, step=25)
    st.info(f"Marlas: {side_sqyd/25:.2f} | Kanals: {side_sqyd/500:.4f}")
    if st.button("🚪 LOGOUT"):
        st.session_state.auth_status = False
        st.rerun()

# --- 6. MAIN PREDICTOR LAYOUT ---
l_col, r_col = st.columns([2, 1])

with l_col:
    st.subheader("Property Specifications")
    loc_name = st.selectbox("Location / Sector", locations)
    c1, c2, c3, c4 = st.columns(4)
    area = c1.number_input("Area (SqYd)", 1, 10000, 125)
    beds = c2.number_input("Beds", 1, 10, 3)
    baths = c3.number_input("Baths", 1, 10, 3)
    kitchens = c4.number_input("Kitchens", 1, 5, 1)
    predict_btn = st.button("🚀 GENERATE HYBRID VALUATION")

with r_col:
    geolocator = Nominatim(user_agent="ZameenAI_App")
    try:
        location_data = geolocator.geocode(f"{loc_name}, Pakistan")
        if location_data:
            st.map(pd.DataFrame({'lat': [location_data.latitude], 'lon': [location_data.longitude]}))
    except: st.write("Map loading...")

# --- 7. THE FUNCTIONAL FIX ---
if predict_btn:
    if model:
        try:
            # We pass ONLY the raw columns. The pipeline handles the expansion to 250 features.
            input_df = pd.DataFrame({
                'Location': [loc_name],
                'Area': [area],
                'Baths': [baths],
                'Beds': [beds],
                'Kitchens': [kitchens],
                'Drawing Room': [1],        
                'Lounge or Sitting Room': [1] 
            })

            # The model pipeline now correctly receives all 250 expected features
            log_prediction = model.predict(input_df)[0]
            final_price = np.expm1(log_prediction)

            st.balloons()
            st.markdown(f"""
                <div class="price-card">
                    <h3 style="color:#10b981; margin:0;">AI ESTIMATED VALUE</h3>
                    <h1 style="color:white; margin:0;">PKR {int(final_price):,}</h1>
                </div>
            """, unsafe_allow_html=True)
            
            add_history(st.session_state.username, loc_name, area, final_price, "Stable")
        except Exception as e:
            st.error(f"Prediction Error: {e}")
