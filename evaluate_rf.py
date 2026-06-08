"""
Random Forest / Extra Trees Ensemble Evaluation
===============================================
Evals the tree-based ensemble and generates metrics plots.
"""

import os
import json
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from evaluate import plot_parity, plot_metrics_bar
from data_preprocessing import TARGET_NAMES

def load_rf_ensemble(base_dir):
    save_dir = os.path.join(base_dir, 'saved_models')

    with open(os.path.join(save_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    # Load ExtraTrees models natively
    models = joblib.load(os.path.join(save_dir, 'rf_models.pkl'))

    feature_scaler = joblib.load(os.path.join(save_dir, 'feature_scaler.pkl'))
    target_scalers = joblib.load(os.path.join(save_dir, 'target_scalers.pkl'))
    X_test_raw = np.load(os.path.join(save_dir, 'X_test_raw.npy'))
    Y_test_raw = np.load(os.path.join(save_dir, 'Y_test_raw.npy'))

    return models, feature_scaler, target_scalers, X_test_raw, Y_test_raw, metadata

def ensemble_predict_rf(models, X_test_raw, feature_scaler, target_scalers):
    X_scaled = feature_scaler.transform(X_test_raw)

    preds_scaled = np.zeros((X_scaled.shape[0], len(models)))
    for i, model in enumerate(models):
        preds_scaled[:, i] = model.predict(X_scaled)

    preds_original = np.zeros_like(preds_scaled)
    for i, scaler in enumerate(target_scalers):
        if hasattr(scaler, 'center_') and scaler.center_ is not None:
            preds_original[:, i] = scaler.inverse_transform(
                preds_scaled[:, i].reshape(-1, 1)).ravel()
        elif hasattr(scaler, 'mean_') and scaler.mean_ is not None:
            preds_original[:, i] = scaler.inverse_transform(
                preds_scaled[:, i].reshape(-1, 1)).ravel()
        else:
            preds_original[:, i] = preds_scaled[:, i]

    return preds_original

def evaluate_rf():
    print("\n" + "█" * 60)
    print("  MPEA Multi-Output Regression — ExtraTrees Evaluation")
    print("█" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results')
    models, feature_scaler, target_scalers, X_test_raw, Y_test_raw, metadata = \
        load_rf_ensemble(base_dir)

    Y_pred = ensemble_predict_rf(models, X_test_raw, feature_scaler, target_scalers)

    print(f"\n{'='*65}")
    print(f"  TEST SET METRICS (Original Scale, ExtraTrees {len(models)} targets)")
    print(f"{'='*65}")
    print(f"\n{'Target':<25} {'R²':>8} {'MAE':>10} {'RMSE':>10} {'N':>6}")
    print("─" * 65)
    
    r2_list = []
    test_metrics = {}
    for i, name in enumerate(TARGET_NAMES):
        valid = ~np.isnan(Y_test_raw[:, i])
        if valid.sum() < 5:
            continue
        true_i, pred_i = Y_test_raw[valid, i], Y_pred[valid, i]
        r2 = r2_score(true_i, pred_i)
        mae = mean_absolute_error(true_i, pred_i)
        rmse = np.sqrt(mean_squared_error(true_i, pred_i))
        r2_list.append(r2)
        test_metrics[name] = {'R2': float(r2), 'MAE': float(mae), 'RMSE': float(rmse)}
        print(f"{name:<25} {r2:>8.4f} {mae:>10.2f} {rmse:>10.2f} {valid.sum():>6}")
    
    overall_r2 = np.mean(r2_list)
    print(f"\n{'OVERALL AVG R²':<25} {overall_r2:>8.4f}")

    print(f"\nGenerating plots...")
    plot_parity(Y_test_raw, Y_pred, results_dir)
    plot_metrics_bar(Y_test_raw, Y_pred, results_dir)
    
    # Save the history summary so UI / JSON readers don't breaking
    with open(os.path.join(results_dir, 'training_history_rf.json'), 'w') as f:
        json.dump({'test_metrics': test_metrics, 'overall_avg_r2': overall_r2}, f, indent=2)

    print(f"\n✓ All ExtraTrees plots saved to: {results_dir}/")

if __name__ == '__main__':
    evaluate_rf()
