from flask import Flask, request, render_template, jsonify
import sys
import os
import pandas as pd
from werkzeug.utils import secure_filename

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from predict import load_artifacts, preprocess_input, predict_sample
import joblib
import numpy as np

app = Flask(__name__)

# Upload folder config
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load artifacts once at startup
rf, iso, scaler, le_dict, selector = load_artifacts()

# Feature names (10 features)
FEATURE_NAMES =['service', 'flag', 'src_bytes', 'count', 'same_srv_rate', 'diff_srv_rate', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_serror_rate']

# Field types and valid values for validation
FIELD_INFO = {
    'service': {
        'type': 'categorical',
        'placeholder': 'e.g., http, ftp_data, smtp',
        'examples': ['http', 'ftp_data', 'smtp', 'pop_3', 'finger'],
        'error': 'Service must be a valid protocol name (e.g., http, ftp_data)'
    },
    'flag': {
        'type': 'categorical',
        'placeholder': 'e.g., SF, S0, REJ (UPPERCASE)',
        'examples': ['SF', 'S0', 'REJ', 'RSTO', 'RSTOS0'],
        'error': 'Flag must be UPPERCASE (e.g., SF, S0, REJ)'
    },
    'src_bytes': {
        'type': 'numeric',
        'placeholder': 'e.g., 0, 491, 1500',
        'error': 'Source bytes must be a number'
    },
    'count': {
        'type': 'numeric',
        'placeholder': 'e.g., 1, 10, 100',
        'error': 'Count must be a number'
    },
    'same_srv_rate': {
        'type': 'numeric',
        'placeholder': 'e.g., 0.0 to 1.0',
        'error': 'Same service rate must be a number between 0.0 and 1.0'
    },
    'diff_srv_rate': {
        'type': 'numeric',
        'placeholder': 'e.g., 0.0 to 1.0',
        'error': 'Different service rate must be a number between 0.0 and 1.0'
    },
    'dst_host_srv_count': {
        'type': 'numeric',
        'placeholder': 'e.g., 1, 25, 255',
        'error': 'Destination host service count must be a number'
    },
    'dst_host_same_srv_rate': {
        'type': 'numeric',
        'placeholder': 'e.g., 0.0 to 1.0',
        'error': 'Destination host same service rate must be a number between 0.0 and 1.0'
    },
    'dst_host_diff_srv_rate': {
        'type': 'numeric',
        'placeholder': 'e.g., 0.0 to 1.0',
        'error': 'Destination host different service rate must be a number between 0.0 and 1.0'
    },
    'dst_host_serror_rate': {
        'type': 'numeric',
        'placeholder': 'e.g., 0.0 to 1.0',
        'error': 'Destination host SYN error rate must be a number between 0.0 and 1.0'
    }
}

def validate_input(input_data):
    errors = []
    
    for field in FEATURE_NAMES:
        if field not in input_data or str(input_data[field]).strip() == '':
            errors.append(f"'{field}' is required")
            continue
        
        value = input_data[field]
        field_info = FIELD_INFO[field]
        
        if field_info['type'] == 'numeric':
            try:
                num_val = float(value)
                if 'rate' in field and (num_val < 0 or num_val > 1):
                    errors.append(f"'{field}' must be between 0.0 and 1.0 (got {num_val})")
            except ValueError:
                errors.append(f"'{field}' must be a number (got '{value}')")
        
        elif field_info['type'] == 'categorical':
            if field == 'flag':
                if not str(value).isupper():
                    errors.append(f"'{field}' must be UPPERCASE (e.g., SF, S0, REJ) - got '{value}'")
                valid_flags = ['SF', 'S0', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'SH']
                if value.upper() not in valid_flags:
                    errors.append(f"'{field}' '{value}' may not be a valid flag. Use: SF, S0, REJ, RSTO, RSTOS0, RSTR, SH")
    
    return errors

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if this is a file upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return render_template('index.html', error="No file selected", features=FEATURE_NAMES, field_info=FIELD_INFO)
            if file and allowed_file(file.filename):
                try:
                    df = pd.read_csv(file)
                    if df.shape[0] != 1:
                        return render_template('index.html', error="File must contain exactly one row (single sample)", features=FEATURE_NAMES, field_info=FIELD_INFO)
                    
                    raw_input = {}
                    for feat in FEATURE_NAMES:
                        if feat in df.columns:
                            raw_input[feat] = df.iloc[0][feat]
                        else:
                            return render_template('index.html', error=f"Missing required column: '{feat}'", features=FEATURE_NAMES, field_info=FIELD_INFO)
                    
                    validation_errors = validate_input(raw_input)
                    if validation_errors:
                        error_msg = "Validation errors:<br>" + "<br>".join(validation_errors)
                        return render_template('index.html', error=error_msg, features=FEATURE_NAMES, field_info=FIELD_INFO)
                    
                    pred, confidence, pred_type = predict_sample(raw_input)
                    return render_template('result.html', 
                                           prediction=pred, 
                                           confidence=confidence,
                                           pred_type=pred_type,
                                           input_data=raw_input)
                except Exception as e:
                    return render_template('index.html', error=f"Error processing CSV: {str(e)}", features=FEATURE_NAMES, field_info=FIELD_INFO)
        else:
            # Manual form submission
            raw_input = {}
            for feat in FEATURE_NAMES:
                val = request.form.get(feat, '').strip()
                raw_input[feat] = val
            
            validation_errors = validate_input(raw_input)
            if validation_errors:
                error_msg = "Validation errors:<br>" + "<br>".join(validation_errors)
                return render_template('index.html', error=error_msg, features=FEATURE_NAMES, field_info=FIELD_INFO)
            
            try:
                pred, confidence, pred_type = predict_sample(raw_input)
                return render_template('result.html', 
                                       prediction=pred, 
                                       confidence=confidence,
                                       pred_type=pred_type,
                                       input_data=raw_input)
            except Exception as e:
                return render_template('index.html', error=f"Prediction error: {str(e)}", features=FEATURE_NAMES, field_info=FIELD_INFO)
    
    # ✅ FIXED: Always pass field_info to the template
    return render_template('index.html', features=FEATURE_NAMES, field_info=FIELD_INFO, error=None)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)