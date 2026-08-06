import streamlit as st
import pandas as pd
import sys
import os
import time
import requests

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.predict import predict_sample

st.set_page_config(
    page_title="Hybrid NIDS",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

hide_streamlit_style = """
    <style>
        header[data-testid="stHeader"] { display: none !important; }
        .stAppViewContainer { padding-top: 0 !important; }
        .block-container { padding-top: 0 !important; padding-bottom: 1rem !important; }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display: none;}
        .stSidebar {display: none;}
        section[data-testid="stSidebar"] {display: none !important;}
        .stApp {margin-top: 0 !important;}
        .st-emotion-cache-1v0mbdj {padding-top: 0 !important;}
        .st-emotion-cache-18ni7ap {padding-top: 0 !important;}
        .stApp > div:first-child {background: transparent !important;}
        .st-emotion-cache-1r6slb0 {background: transparent !important;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- Custom CSS ---
st.markdown("""
<style>
    body, .stApp {
        background-color: #0a0e17;
        background-image: radial-gradient(circle at 20% 50%, rgba(0, 150, 255, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(100, 0, 255, 0.05) 0%, transparent 50%),
            repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,0.01) 2px, rgba(255,255,255,0.01) 4px),
            repeating-linear-gradient(90deg, transparent, transparent 2px, rgba(255,255,255,0.01) 2px, rgba(255,255,255,0.01) 4px);
        background-size: cover, cover, 20px 20px, 20px 20px;
        min-height: 100vh;
        color: #e0e0e0;
    }
    .container {
        background: rgba(10, 14, 23, 0.85);
        backdrop-filter: blur(12px);
        border-radius: 24px;
        padding: 30px 40px;
        max-width: 1100px;
        margin: 20px auto;
        border: 1px solid rgba(0, 150, 255, 0.2);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.8);
    }
    .header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 25px;
        padding-bottom: 15px;
        border-bottom: 1px solid rgba(0, 150, 255, 0.15);
    }
    .header h1 {
        font-size: 26px;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4ff, #0066ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
    }
    .header .badge {
        background: rgba(0, 150, 255, 0.15);
        color: #00d4ff;
        padding: 4px 14px;
        border-radius: 30px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid rgba(0, 150, 255, 0.3);
    }
    .result-box {
        background: rgba(255,255,255,0.03);
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        border: 2px solid #00d4ff;
        text-align: center;
        box-shadow: 0 0 30px rgba(0, 150, 255, 0.1);
    }
    .prediction { font-size: 48px; font-weight: 900; letter-spacing: 1px; margin-bottom: 5px; text-shadow: 0 0 20px currentColor; }
    .prediction.known { color: #ff6b6b; }
    .prediction.anomaly { color: #ff9f43; }
    .prediction.normal { color: #2ecc71; }
    .confidence { font-size: 20px; color: #b0c4d8; }
    .type-badge {
        display: inline-block; padding: 4px 20px; border-radius: 30px; font-size: 14px; font-weight: 600;
        margin-top: 12px; background: rgba(255,255,255,0.05); color: #b0c4d8; border: 1px solid rgba(255,255,255,0.1);
    }
    .stForm button[type="submit"] {
        background: linear-gradient(135deg, #00d4ff, #0066ff) !important;
        border: none !important; padding: 8px 32px !important; border-radius: 30px !important;
        font-size: 15px !important; font-weight: 600 !important; color: white !important;
        box-shadow: 0 0 20px rgba(0, 150, 255, 0.25) !important; transition: all 0.2s ease !important; cursor: pointer !important;
        display: inline-block !important; margin: 0 auto !important;
    }
    .stForm button[type="submit"]:hover { transform: scale(1.02) !important; box-shadow: 0 0 30px rgba(0, 150, 255, 0.4) !important; }
    div[data-testid="stForm"] > div:last-child { display: flex !important; justify-content: center !important; padding-top: 5px !important; width: 100% !important; }
    input[type="number"]::-webkit-outer-spin-button, input[type="number"]::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }
    input[type="number"] { -moz-appearance: textfield; appearance: textfield; }
    .stSelectbox div[data-baseweb="select"] { background: rgba(255,255,255,0.05) !important; border-radius: 8px !important; border: 1px solid rgba(255,255,255,0.1) !important; }
    .stSelectbox div[data-baseweb="select"]:focus-within { border-color: #00d4ff !important; box-shadow: 0 0 0 3px rgba(0, 150, 255, 0.2) !important; }
    .stSelectbox div[data-baseweb="select"] div[role="button"] { color: #8aa3c0 !important; }
    .stSelectbox div[data-baseweb="select"] div[role="button"] div[data-value] { color: #e0e0e0 !important; }
</style>
""", unsafe_allow_html=True)

# --- Feature definitions ---
FEATURES_INFO = {
    'service': {'label': 'service', 'type': 'select', 'options': ['http', 'ftp_data', 'smtp', 'pop_3', 'telnet', 'finger', 'ftp', 'domain', 'ssh', 'gopher', 'mtp', 'netbios_ssn', 'other'], 'placeholder': 'Select service...', 'hint': 'The service running on the destination port'},
    'flag': {'label': 'flag', 'type': 'select', 'options': ['SF', 'S0', 'REJ', 'RSTO', 'RSTR', 'SH', 'S1', 'S2', 'S3'], 'placeholder': 'Select flag...', 'hint': 'TCP handshake status'},
    'src_bytes': {'label': 'src_bytes', 'type': 'number', 'default': 0.0, 'step': 1.0, 'format': '%.0f', 'hint': 'Source bytes'},
    'count': {'label': 'count', 'type': 'number', 'default': 1, 'step': 1.0, 'format': '%.0f', 'hint': 'Connections in last 2s'},
    'same_srv_rate': {'label': 'same_srv_rate', 'type': 'number', 'default': 0.5, 'step': 0.01, 'format': '%.4f', 'hint': 'Same service proportion'},
    'diff_srv_rate': {'label': 'diff_srv_rate', 'type': 'number', 'default': 0.0, 'step': 0.01, 'format': '%.4f', 'hint': 'Diff service proportion'},
    'dst_host_srv_count': {'label': 'dst_host_srv_count', 'type': 'number', 'default': 1, 'step': 1.0, 'format': '%.0f', 'hint': 'Dst host same service connections'},
    'dst_host_same_srv_rate': {'label': 'dst_host_same_srv_rate', 'type': 'number', 'default': 0.0, 'step': 0.01, 'format': '%.4f', 'hint': 'Dst host same service proportion'},
    'dst_host_diff_srv_rate': {'label': 'dst_host_diff_srv_rate', 'type': 'number', 'default': 0.0, 'step': 0.01, 'format': '%.4f', 'hint': 'Dst host diff service proportion'},
    'dst_host_serror_rate': {'label': 'dst_host_serror_rate', 'type': 'number', 'default': 0.0, 'step': 0.01, 'format': '%.4f', 'hint': 'Dst host SYN error proportion'}
}
FEATURES = list(FEATURES_INFO.keys())

# --- Main UI ---
st.markdown('<div class="container">', unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header">
    <h1>🔒 Hybrid NIDS</h1>
    <span class="badge">Zero‑Day Ready</span>
</div>
""", unsafe_allow_html=True)

# New Mode Selector in Sidebar (accessible even though we hid the sidebar element using CSS)
st.sidebar.title("🕹️ Control Panel")
mode = st.sidebar.radio("Select Input Mode", ["✍️ Manual Input", "🌐 Live Demo"])

# --- MODULE 1: MANUAL INPUT ---
error_placeholder = st.empty()
if mode == "✍️ Manual Input":
    with st.form(key="manual_form", clear_on_submit=False):
        cols = st.columns(2)
        input_data = {}
        for i, feat in enumerate(FEATURES):
            col = cols[i % 2]
            info = FEATURES_INFO[feat]
            if info['type'] == 'select':
                val = col.selectbox(label=info['label'], options=info['options'], index=None, placeholder=info['placeholder'], help=info['hint'])
                if val is None: val = info['options'][0]
            else:
                val = col.number_input(label=info['label'], value=float(info['default']), step=info['step'], format=info['format'], help=info['hint'])
            input_data[feat] = val
        submitted = st.form_submit_button("🔍 Analyze")
    
    if submitted:
        with st.spinner("Running prediction..."):
            try:
                pred, conf, ptype = predict_sample(input_data)
                error_placeholder.empty()
                color_map = {'known': '#ff6b6b' if pred != 'normal' else '#2ecc71', 'anomaly': '#ff9f43', 'normal': '#00d4ff'}
                label_map = {'known': '🔴 Known Attack' if pred != 'normal' else '🟢 Normal Traffic', 'anomaly': '🟠 Zero‑Day Anomaly', 'normal': '🔵 Low Confidence Normal'}
                color = color_map.get(ptype, '#00d4ff')
                label = label_map.get(ptype, '')
                st.markdown(f"""
                <div class="result-box" style="border-color: {color};">
                    <div style="font-size: 14px; color: #8aa3c0; font-weight: 600; letter-spacing: 2px; text-transform: uppercase;">🎯 Prediction Result</div>
                    <div class="prediction {ptype}" style="color: {color};">{pred}</div>
                    <div class="confidence">Confidence: <strong>{conf:.4f}</strong></div>
                    <div class="type-badge" style="background: {color}22; color: {color}; border-color: {color}44;">{label}</div>
                </div>
                """, unsafe_allow_html=True)
                st.subheader("📊 Input Features")
                st.dataframe(pd.DataFrame([input_data]), use_container_width=True, hide_index=True)
            except Exception as e:
                error_placeholder.markdown(f'<div style="color:#ff6b6b; background: rgba(255,0,0,0.1); padding: 10px; border-radius: 10px;">Prediction error: {str(e)}</div>', unsafe_allow_html=True)

# --- MODULE 2: LIVE DEMO ---
elif mode == "🌐 Live Demo":
    st.markdown("<h4>🟢 Live Packet Capture Mode</h4>", unsafe_allow_html=True)
    st.info("Make sure your Local Sniffer agent and Bridge server are running, and ngrok is active.")
    
    # IMPORTANT: REPLACE THIS WITH YOUR NGROK URL (e.g., https://xxxx.ngrok-free.app)
    BRIDGE_URL = "https://xxxx.ngrok-free.app/get_features" 
    
    col1, col2 = st.columns([8, 2])
    with col2:
        if st.button("🛑 Stop Monitoring"):
            st.session_state['live_active'] = False
        else:
            if 'live_active' not in st.session_state:
                st.session_state['live_active'] = True

    if st.session_state['live_active']:
        result_container = st.empty()
        try:
            res = requests.get(BRIDGE_URL)
            if res.status_code == 200:
                features = res.json()
                if features:
                    pred, conf, ptype = predict_sample(features)
                    color_map = {'known': '#ff6b6b' if pred != 'normal' else '#2ecc71', 'anomaly': '#ff9f43', 'normal': '#00d4ff'}
                    label_map = {'known': '🔴 Known Attack' if pred != 'normal' else '🟢 Normal Traffic', 'anomaly': '🟠 Zero‑Day Anomaly', 'normal': '🔵 Low Confidence Normal'}
                    color = color_map.get(ptype, '#00d4ff')
                    label = label_map.get(ptype, '')
                    result_container.markdown(f"""
                    <div class="result-box" style="border-color: {color};">
                        <div style="font-size: 14px; color: #8aa3c0; font-weight: 600; letter-spacing: 2px; text-transform: uppercase;">📡 Live Packet Detected</div>
                        <div class="prediction {ptype}" style="color: {color};">{pred}</div>
                        <div class="confidence">Confidence: <strong>{conf:.4f}</strong></div>
                        <div class="type-badge" style="background: {color}22; color: {color}; border-color: {color}44;">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)
        except Exception as e:
            result_container.warning(f"⚠️ Bridge connection error: {e}")

        # Wait 2 seconds and reload the page to check for new data
        time.sleep(2)
        st.rerun()
    else:
        st.write("Live monitoring has been stopped.")

# Footer
st.markdown("""
<div class="footer">
    &copy; 2026 Hybrid NIDS | Powered by Machine Learning
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
