import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.predict import predict_sample

st.set_page_config(
    page_title="Hybrid NIDS",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Hide Streamlit default elements ---
hide_streamlit_style = """
    <style>
        header {display: none !important;}
        #MainMenu {display: none !important;}
        footer {display: none !important;}
        .stApp > header {display: none !important;}
        .stDeployButton {display: none !important;}
        .stSidebar {display: none !important;}
        section[data-testid="stSidebar"] {display: none !important;}
        .st-emotion-cache-1v0mbdj {padding-top: 0 !important;}
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
        }
        .stApp > div:first-child {background: transparent !important;}
        .st-emotion-cache-1r6slb0 {background: transparent !important;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- JavaScript for Enter-to-Next ---
js_nav = """
<script>
(function() {
    function attachEnterListeners() {
        const form = document.querySelector('form[data-testid="stForm"]');
        if (!form) return;
        const inputs = form.querySelectorAll('input, select');
        const submitBtn = form.querySelector('button[type="submit"]');
        if (inputs.length === 0) return;

        inputs.forEach((input, index) => {
            if (input.dataset.enterListener) return;
            input.dataset.enterListener = 'true';

            input.addEventListener('keydown', function(e) {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    if (index === inputs.length - 1) {
                        if (submitBtn) submitBtn.click();
                    } else {
                        const next = inputs[index + 1];
                        if (next) {
                            next.focus();
                            if (next.tagName === 'SELECT') {
                                next.click();
                            }
                        }
                    }
                }
            });
        });
    }

    attachEnterListeners();
    const observer = new MutationObserver(function() {
        attachEnterListeners();
    });
    observer.observe(document.body, { childList: true, subtree: true });
})();
</script>
"""
st.markdown(js_nav, unsafe_allow_html=True)

# --- CSS Styling (Dark theme, NO container box) ---
st.markdown("""
<style>
    body, .stApp {
        background-color: #0a0e17;
        background-image: 
            radial-gradient(circle at 20% 50%, rgba(0, 150, 255, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(100, 0, 255, 0.05) 0%, transparent 50%),
            repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,0.01) 2px, rgba(255,255,255,0.01) 4px),
            repeating-linear-gradient(90deg, transparent, transparent 2px, rgba(255,255,255,0.01) 2px, rgba(255,255,255,0.01) 4px);
        background-size: cover, cover, 20px 20px, 20px 20px;
        min-height: 100vh;
        color: #e0e0e0;
        margin: 0;
        padding: 0;
    }
    /* Hide + / - on number inputs */
    input[type="number"]::-webkit-outer-spin-button,
    input[type="number"]::-webkit-inner-spin-button {
        -webkit-appearance: none;
        margin: 0;
    }
    input[type="number"] {
        -moz-appearance: textfield;
        appearance: textfield;
    }

    /* ✅ Container is now TRANSPARENT – no box at all */
    .container {
        background: transparent !important;
        backdrop-filter: none !important;
        border: none !important;
        box-shadow: none !important;
        padding: 20px 30px !important;
        max-width: 1100px;
        margin: 0 auto;
        width: 100%;
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
        text-shadow: 0 0 30px rgba(0, 150, 255, 0.3);
    }
    .header .badge {
        background: rgba(0, 150, 255, 0.15);
        color: #00d4ff;
        padding: 4px 14px;
        border-radius: 30px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid rgba(0, 150, 255, 0.3);
        box-shadow: 0 0 20px rgba(0, 150, 255, 0.1);
    }
    .subtitle {
        color: #a0b4c8;
        margin-bottom: 20px;
        font-size: 15px;
        line-height: 1.5;
    }
    .form-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
        gap: 14px 18px;
        margin-bottom: 20px;
    }
    .form-item label {
        font-size: 12px;
        font-weight: 600;
        color: #8aa3c0;
        margin-bottom: 2px;
    }
    .form-item input, .form-item select {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 8px;
        padding: 6px 10px;
        font-size: 13px;
        color: #e0e0e0;
        width: 100%;
    }
    .form-item input:focus, .form-item select:focus {
        border-color: #00d4ff;
        box-shadow: 0 0 0 3px rgba(0, 150, 255, 0.2);
        outline: none;
    }
    .form-item input::placeholder {
        color: #5a6e85;
    }
    .hint {
        font-size: 10px;
        color: #5a6e85;
        margin-top: 1px;
    }
    /* Small Analyze button */
    .stForm button[type="submit"] {
        background: linear-gradient(135deg, #00d4ff, #0066ff) !important;
        border: none !important;
        padding: 8px 32px !important;
        border-radius: 30px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        color: white !important;
        box-shadow: 0 0 20px rgba(0, 150, 255, 0.25) !important;
        transition: all 0.2s ease !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        cursor: pointer !important;
        display: inline-block !important;
        margin: 0 auto !important;
    }
    .stForm button[type="submit"]:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 0 30px rgba(0, 150, 255, 0.4) !important;
    }
    div[data-testid="stForm"] > div:last-child {
        display: flex !important;
        justify-content: center !important;
        padding-top: 5px !important;
        width: 100% !important;
    }
    .result-box {
        background: rgba(255,255,255,0.05);
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        border: 2px solid #00d4ff;
        text-align: center;
        box-shadow: 0 0 30px rgba(0, 150, 255, 0.1);
    }
    .prediction {
        font-size: 48px;
        font-weight: 900;
        letter-spacing: 1px;
        margin-bottom: 5px;
        text-shadow: 0 0 20px currentColor;
    }
    .prediction.known { color: #ff6b6b; }
    .prediction.anomaly { color: #ff9f43; }
    .prediction.normal { color: #2ecc71; }
    .confidence {
        font-size: 20px;
        color: #b0c4d8;
    }
    .type-badge {
        display: inline-block;
        padding: 4px 20px;
        border-radius: 30px;
        font-size: 14px;
        font-weight: 600;
        margin-top: 12px;
        background: rgba(255,255,255,0.05);
        color: #b0c4d8;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .table-wrap {
        overflow-x: auto;
        margin: 20px 0;
        background: rgba(255,255,255,0.02);
        border-radius: 12px;
        padding: 5px;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        color: #c0d0e0;
    }
    th {
        background: rgba(0, 150, 255, 0.1);
        padding: 8px 10px;
        text-align: left;
        font-weight: 600;
        color: #80b0d8;
        border-bottom: 1px solid rgba(0,150,255,0.2);
    }
    td {
        padding: 6px 10px;
        border-bottom: 1px solid rgba(255,255,255,0.03);
    }
    tr:hover td {
        background: rgba(0, 150, 255, 0.04);
    }
    .footer {
        margin-top: 30px;
        text-align: center;
        color: #5a6e85;
        font-size: 13px;
        border-top: 1px solid rgba(255,255,255,0.05);
        padding-top: 15px;
    }
    .error {
        background: rgba(255, 0, 0, 0.1);
        color: #ff6b6b;
        padding: 10px 16px;
        border-radius: 10px;
        border-left: 4px solid #ff6b6b;
        margin-bottom: 15px;
        border: 1px solid rgba(255,0,0,0.2);
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# --- Feature definitions (unchanged) ---
FEATURES_INFO = {
    'service': {'label': 'service', 'placeholder': 'e.g., http, ftp_data', 'type': 'text', 'default': 'http', 'hint': 'Examples: http, ftp_data, smtp'},
    'flag': {'label': 'flag', 'placeholder': 'e.g., SF, S0, REJ (UPPERCASE)', 'type': 'text', 'default': 'SF', 'hint': 'Examples: SF, S0, REJ, RSTO'},
    'src_bytes': {'label': 'src_bytes', 'placeholder': 'e.g., 0, 491', 'type': 'number', 'default': 0.0, 'hint': 'Source bytes (0–1e9)'},
    'count': {'label': 'count', 'placeholder': 'e.g., 1, 10', 'type': 'number', 'default': 1, 'hint': 'Number of connections in window'},
    'same_srv_rate': {'label': 'same_srv_rate', 'placeholder': 'e.g., 0.0 to 1.0', 'type': 'number', 'default': 0.5, 'hint': 'Proportion to same service'},
    'diff_srv_rate': {'label': 'diff_srv_rate', 'placeholder': 'e.g., 0.0 to 1.0', 'type': 'number', 'default': 0.0, 'hint': 'Proportion to different services'},
    'dst_host_srv_count': {'label': 'dst_host_srv_count', 'placeholder': 'e.g., 1, 25', 'type': 'number', 'default': 1, 'hint': 'Connections to destination host on same service'},
    'dst_host_same_srv_rate': {'label': 'dst_host_same_srv_rate', 'placeholder': 'e.g., 0.0 to 1.0', 'type': 'number', 'default': 0.0, 'hint': 'Proportion to destination host same service'},
    'dst_host_diff_srv_rate': {'label': 'dst_host_diff_srv_rate', 'placeholder': 'e.g., 0.0 to 1.0', 'type': 'number', 'default': 0.0, 'hint': 'Proportion to destination host different service'},
    'dst_host_serror_rate': {'label': 'dst_host_serror_rate', 'placeholder': 'e.g., 0.0 to 1.0', 'type': 'number', 'default': 0.0, 'hint': 'Proportion of SYN errors to destination host'}
}
FEATURES = list(FEATURES_INFO.keys())

# --- Main UI ---
st.markdown('<div class="container">', unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <h1>🔒 Hybrid NIDS</h1>
    <span class="badge">Zero‑Day Ready</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<p class="subtitle">Enter the network flow features manually. Press <strong>Enter</strong> to go to the next field; on the last field, Enter submits.</p>', unsafe_allow_html=True)

error_placeholder = st.empty()

with st.form(key="manual_form", clear_on_submit=False):
    cols = st.columns(2)
    input_data = {}
    for i, feat in enumerate(FEATURES):
        col = cols[i % 2]
        info = FEATURES_INFO[feat]
        if info['type'] == 'text':
            val = col.text_input(label=info['label'], value=info['default'], placeholder=info['placeholder'], help=info['hint'])
        else:
            val = col.number_input(label=info['label'], value=float(info['default']),
                                   step=0.01 if 'rate' in feat or 'src_bytes' in feat else 1.0,
                                   format="%.4f" if 'rate' in feat else "%.0f",
                                   help=info['hint'])
        input_data[feat] = val

    submitted = st.form_submit_button("🔍 Analyze")

# --- Handle prediction ---
if submitted:
    with st.spinner("Running prediction..."):
        try:
            pred, conf, ptype = predict_sample(input_data)
            error_placeholder.empty()

            color_map = {
                'known': '#ff6b6b' if pred != 'normal' else '#2ecc71',
                'anomaly': '#ff9f43',
                'normal': '#00d4ff'
            }
            label_map = {
                'known': '🔴 Known Attack' if pred != 'normal' else '🟢 Normal Traffic',
                'anomaly': '🟠 Zero‑Day Anomaly',
                'normal': '🔵 Low Confidence Normal'
            }
            color = color_map.get(ptype, '#00d4ff')
            label = label_map.get(ptype, '')

            st.markdown(f"""
            <div class="result-box" style="border-color: {color};">
                <div style="font-size: 14px; color: #8aa3c0; font-weight: 600; letter-spacing: 2px; text-transform: uppercase;">
                    🎯 Prediction Result
                </div>
                <div class="prediction {ptype}" style="color: {color};">
                    {pred}
                </div>
                <div class="confidence">
                    Confidence: <strong>{conf:.4f}</strong>
                </div>
                <div class="type-badge" style="background: {color}22; color: {color}; border-color: {color}44;">
                    {label}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.subheader("📊 Input Features")
            df_input = pd.DataFrame([input_data])
            st.dataframe(df_input, use_container_width=True, hide_index=True)

        except Exception as e:
            error_placeholder.markdown(f'<div class="error">Prediction error: {str(e)}</div>', unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    &copy; 2026 Hybrid NIDS | Powered by Machine Learning
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)