"""
Flask Web Application for MPEA Property Prediction
=====================================================
Modern web interface for real-time prediction of mechanical properties
from alloy compositions.
"""

import os
import json
import numpy as np
import joblib

from train_hybrid_admm_dl import ADMMLassoRegressor
import __main__
__main__.ADMMLassoRegressor = ADMMLassoRegressor

from flask import Flask, render_template, request, jsonify

from data_preprocessing import ALL_ELEMENTS, TARGET_NAMES, parse_formula, compute_domain_features

app = Flask(__name__)

import torch
from model import MultiOutputRegressor

# ─── Global model and scalers ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, 'saved_models')

model_dl = None
rf_models = None
admm_models = None
feature_scaler = None
target_scalers = None
metadata = None
category_mappings = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model():
    """Load the trained PINN PyTorch model and all artifacts."""
    global model_dl, rf_models, admm_models, feature_scaler, target_scalers, metadata, category_mappings

    # Load metadata
    with open(os.path.join(SAVE_DIR, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    with open(os.path.join(SAVE_DIR, 'category_mappings.json'), 'r') as f:
        category_mappings = json.load(f)

    # Load PyTorch PINN model natively
    model_dl = MultiOutputRegressor(n_features=metadata['n_features'], n_targets=metadata['n_targets']).to(device)
    checkpoint = torch.load(os.path.join(SAVE_DIR, 'best_model.pth'), map_location=device)
    
    # Handle state dict wrapping
    if 'model_state_dict' in checkpoint:
        model_dl.load_state_dict(checkpoint['model_state_dict'])
    else:
        model_dl.load_state_dict(checkpoint)
        
    model_dl.eval()  # Freeze BatchNorm1d running statistics (resolves batch_size=1 error)
    
    # Selectively activate ONLY Dropout layers for Deep Monte Carlo uncertainty estimation!
    def apply_dropout(m):
        if type(m) == torch.nn.Dropout:
            m.train()
    model_dl.apply(apply_dropout)

    # Load scalers
    feature_scaler = joblib.load(os.path.join(SAVE_DIR, 'feature_scaler.pkl'))
    target_scalers = joblib.load(os.path.join(SAVE_DIR, 'target_scalers.pkl'))
    
    # Load Traditional Machine Learning hybrid components
    rf_models = joblib.load(os.path.join(SAVE_DIR, 'rf_models.pkl'))
    admm_models = joblib.load(os.path.join(SAVE_DIR, 'admm_models.pkl'))

    print(f"[SUCCESS] Publication-Tier Deep PINN Model loaded successfully on {device}")


def build_feature_vector(elements: dict, processing: str, microstructure: str,
                          bcc_fcc: str, grain_size: float, density: float,
                          test_temp: float) -> np.ndarray:
    """
    Build feature vector from user inputs matching the training feature order.
    """
    feature_names = metadata['feature_names']
    features = np.zeros(len(feature_names))

    # Normalize elemental composition
    total = sum(elements.values())
    if total > 0:
        elements = {k: v / total for k, v in elements.items()}

    # Fill elemental compositions
    for elem, frac in elements.items():
        if elem in feature_names:
            idx = feature_names.index(elem)
            features[idx] = frac

    # Fill categorical one-hot features
    for col, columns in category_mappings.items():
        for col_name in columns:
            if col_name in feature_names:
                idx = feature_names.index(col_name)
                # Check if this dummy matches the user input
                if 'Microstru' in col_name and microstructure.upper() in col_name.upper():
                    features[idx] = 1.0
                elif 'Processing' in col_name and processing.upper() in col_name.upper():
                    features[idx] = 1.0
                elif 'BCC' in col_name or 'FCC' in col_name or 'other' in col_name:
                    if bcc_fcc.upper() in col_name.upper():
                        features[idx] = 1.0

    # Fill numeric features
    for i, fname in enumerate(feature_names):
        if 'grain' in fname.lower() and grain_size > 0:
            features[i] = grain_size
        elif 'density' in fname.lower() and density > 0:
            features[i] = density
        elif 'temperature' in fname.lower():
            features[i] = test_temp
            
    # CRITICAL: Fill in materials science domain features
    domain_features = compute_domain_features(elements)
    for fname, val in domain_features.items():
        if fname in feature_names:
            idx = feature_names.index(fname)
            features[idx] = val

    return features.reshape(1, -1)


@app.route('/')
def index():
    """Render the main prediction page."""
    return render_template('index.html', elements=ALL_ELEMENTS, targets=TARGET_NAMES)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        data = request.get_json()

        # Extract element compositions
        elements = {}
        for elem in ALL_ELEMENTS:
            val = float(data.get(elem, 0))
            if val > 0:
                elements[elem] = val

        if not elements:
            return jsonify({'error': 'Please enter at least one element composition'}), 400

        # Extract other features
        processing = data.get('processing', 'CAST')
        microstructure = data.get('microstructure', 'FCC')
        bcc_fcc = data.get('bcc_fcc', 'FCC')
        grain_size = float(data.get('grain_size', 0))
        density = float(data.get('density', 0))
        test_temp = float(data.get('test_temp', 25))

        # Build feature vector
        X = build_feature_vector(elements, processing, microstructure,
                                  bcc_fcc, grain_size, density, test_temp)

        # Scale and predict
        X_scaled = feature_scaler.transform(X)
        # Ensure X_tensor requires_grad=False just in case
        X_tensor = torch.FloatTensor(X_scaled).to(device)

        # ==============================================================================
        # ⭐ MODEL 1 START: DEEP NEURAL NETWORK (DNN) + PINN BACKBONE
        # Executes multi-head attention passes with Monte Carlo Dropout for uncertainty
        # ==============================================================================
        mc_passes = 15
        mc_preds = []
        z_embeddings = [] # Need z_latent for ADMM
        
        with torch.no_grad():
            for _ in range(mc_passes):
                preds = model_dl(X_tensor)
                mc_preds.append(preds.cpu().numpy())
                
                # Extract embeddings for ADMM models
                x_proj = model_dl.input_proj(X_tensor)
                shared = model_dl.backbone(x_proj)
                z_latent = model_dl.attention(shared).cpu().numpy()
                z_embeddings.append(z_latent)
                
        mc_preds = np.array(mc_preds) # Shape: (15, 1, 5)
        # We can average MC dropout predictions to act as DL core:
        dl_pred_scaled = mc_preds.mean(axis=0) # Shape: (1, 5)
        dl_uncertainty = mc_preds.std(axis=0) # Shape: (1, 5)
        z_embed_mean = np.array(z_embeddings).mean(axis=0) # Shape: (1, hidden_dim)

        # ==============================================================================
        # ⭐ MODEL 2 START: EXTRA TREES ENSEMBLE (ET / RF)
        # Anchors boundary variance across all preprocessed features
        # ==============================================================================
        rf_pred_scaled = np.array([m.predict(X_scaled)[0] for m in rf_models]).reshape(1, -1)
        
        # ==============================================================================
        # ⭐ MODEL 3 START: ADMM-LASSO REGRESSOR
        # Applies soft-thresholding L1 sparsity directly on the 128-d latent attention embeddings
        # ==============================================================================
        admm_pred_scaled = np.array([m.predict(z_embed_mean)[0] for m in admm_models]).reshape(1, -1)
        
        # 4. Integrate Hybrid Blend (Stable publication weights)
        blend_weights = {
            'Hardness (HV)': {'DL': 0.65, 'RF': 0.30, 'ADMM': 0.05},
            'Yield Strength (MPa)': {'DL': 0.60, 'RF': 0.30, 'ADMM': 0.10},
            'UTS (MPa)': {'DL': 0.70, 'RF': 0.20, 'ADMM': 0.10},
            'Elongation (%)': {'DL': 0.80, 'RF': 0.10, 'ADMM': 0.10},
            'Young Modulus (GPa)': {'DL': 0.50, 'RF': 0.40, 'ADMM': 0.10}
        }
        
        hybrid_pred_scaled = np.zeros_like(dl_pred_scaled)
        for i, name in enumerate(TARGET_NAMES):
            w = blend_weights[name]
            hybrid_pred_scaled[0, i] = w['DL'] * dl_pred_scaled[0, i] + w['RF'] * rf_pred_scaled[0, i] + w['ADMM'] * admm_pred_scaled[0, i]

        # Inverse transform
        predictions = {}
        for i, (name, scaler) in enumerate(zip(TARGET_NAMES, target_scalers)):
            try:
                val = scaler.inverse_transform(hybrid_pred_scaled[0, i].reshape(1, -1))[0, 0]
                # For standard/robust scalers, variance scales linearly via the scale_ parameter
                try:
                    uncert = dl_uncertainty[0, i] * scaler.scale_[0]
                except:
                    uncert = dl_uncertainty[0, i]
            except Exception:
                val = hybrid_pred_scaled[0, i]
                uncert = dl_uncertainty[0, i]
                
            predictions[name] = f"{round(float(val), 2)} ± {round(float(uncert), 2)}"

        # Build formula string
        formula = ' '.join(f"{e}{v}" for e, v in sorted(elements.items()))

        return jsonify({
            'predictions': predictions,
            'formula': formula,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/metrics')
def metrics():
    """Return test set metrics."""
    results_dir = os.path.join(BASE_DIR, 'results')
    history_path = os.path.join(results_dir, 'training_history_rf.json')

    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        return jsonify(history.get('test_metrics', {}))
    return jsonify({})


if __name__ == '__main__':
    load_model()
    app.run(debug=False, host='0.0.0.0', port=5000)
