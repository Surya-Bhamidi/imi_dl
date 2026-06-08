"""
Enhanced Evaluation with Ensemble Support
============================================
"""

import os
import json
import numpy as np
import torch
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from model import MultiOutputRegressor
from data_preprocessing import TARGET_NAMES

plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = ['#6366f1', '#ec4899', '#14b8a6', '#f59e0b', '#8b5cf6']


def load_ensemble(base_dir):
    """Load all ensemble models."""
    save_dir = os.path.join(base_dir, 'saved_models')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(os.path.join(save_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    # Load all ensemble models
    models = []
    i = 0
    while True:
        path = os.path.join(save_dir, f'model_{i}.pth')
        if not os.path.exists(path):
            break
        model = MultiOutputRegressor(
            n_features=metadata['n_features'],
            n_targets=metadata['n_targets']
        ).to(device)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        models.append(model)
        i += 1

    if not models:
        # Fallback to best_model.pth
        model = MultiOutputRegressor(
            n_features=metadata['n_features'],
            n_targets=metadata['n_targets']
        ).to(device)
        ckpt = torch.load(os.path.join(save_dir, 'best_model.pth'),
                          map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        models = [model]

    feature_scaler = joblib.load(os.path.join(save_dir, 'feature_scaler.pkl'))
    target_scalers = joblib.load(os.path.join(save_dir, 'target_scalers.pkl'))
    X_test_raw = np.load(os.path.join(save_dir, 'X_test_raw.npy'))
    Y_test_raw = np.load(os.path.join(save_dir, 'Y_test_raw.npy'))

    return models, feature_scaler, target_scalers, X_test_raw, Y_test_raw, metadata, device


def ensemble_predict(models, X_test_raw, feature_scaler, target_scalers, device):
    """Ensemble prediction (Normalized Scale)."""
    X_scaled = feature_scaler.transform(X_test_raw)
    X_tensor = torch.FloatTensor(X_scaled).to(device)

    all_preds = []
    for model in models:
        with torch.no_grad():
            pred = model(X_tensor).cpu().numpy()
        all_preds.append(pred)

    preds_scaled = np.mean(all_preds, axis=0)

    preds_original = np.zeros_like(preds_scaled)
    for i, scaler in enumerate(target_scalers):
        if hasattr(scaler, 'inverse_transform'):
            preds_original[:, i] = scaler.inverse_transform(
                preds_scaled[:, i].reshape(-1, 1)).ravel()
        else:
            preds_original[:, i] = preds_scaled[:, i]

    return preds_original


def plot_parity(Y_true, Y_pred, results_dir):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()
    for i, name in enumerate(TARGET_NAMES):
        ax = axes[i]
        valid = ~np.isnan(Y_true[:, i])
        if valid.sum() < 5:
            ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(name, fontsize=13, fontweight='bold')
            continue
        true_i, pred_i = Y_true[valid, i], Y_pred[valid, i]
        r2 = r2_score(true_i, pred_i)
        mae = mean_absolute_error(true_i, pred_i)
        rmse = np.sqrt(mean_squared_error(true_i, pred_i))
        ax.scatter(true_i, pred_i, alpha=0.5, s=25, c=COLORS[i],
                   edgecolors='white', linewidth=0.3)
        lims = [min(true_i.min(), pred_i.min()), max(true_i.max(), pred_i.max())]
        margin = (lims[1] - lims[0]) * 0.05
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, '--', color='#475569', linewidth=1.5, alpha=0.7)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('Actual', fontsize=11); ax.set_ylabel('Predicted', fontsize=11)
        ax.set_title(name, fontsize=13, fontweight='bold')
        textstr = f'R² = {r2:.3f}\nMAE = {mae:.1f}\nRMSE = {rmse:.1f}'
        props = dict(boxstyle='round,pad=0.4', facecolor=COLORS[i], alpha=0.15)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
    axes[-1].axis('off')
    plt.suptitle('Multi-Output Regression — Parity Plots (Ensemble)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'parity_plots.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ Saved parity_plots.png")


def plot_training_curves(results_dir):
    path = os.path.join(results_dir, 'training_history.json')
    if not os.path.exists(path):
        return
    with open(path, 'r') as f:
        history = json.load(f)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs, history['train_loss'], label='Train', color='#6366f1', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], label='Validation', color='#ec4899', linewidth=2)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss', fontweight='bold')
    axes[0].legend(); axes[0].set_yscale('log'); axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history['lr'], color='#14b8a6', linewidth=2)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_curves.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ Saved training_curves.png")


def plot_metrics_bar(Y_true, Y_pred, results_dir):
    data = {'Target': [], 'R²': [], 'MAE': [], 'RMSE': []}
    for i, name in enumerate(TARGET_NAMES):
        valid = ~np.isnan(Y_true[:, i])
        if valid.sum() < 5:
            continue
        true_i, pred_i = Y_true[valid, i], Y_pred[valid, i]
        data['Target'].append(name)
        data['R²'].append(r2_score(true_i, pred_i))
        data['MAE'].append(mean_absolute_error(true_i, pred_i))
        data['RMSE'].append(np.sqrt(mean_squared_error(true_i, pred_i)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    n = len(data['Target'])
    bars = axes[0].barh(data['Target'], data['R²'], color=COLORS[:n])
    axes[0].set_xlabel('R² Score'); axes[0].set_title('R² Score', fontweight='bold')
    axes[0].set_xlim(0, 1)
    for bar, val in zip(bars, data['R²']):
        axes[0].text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                     f'{val:.3f}', va='center', fontsize=10)
    bars = axes[1].barh(data['Target'], data['MAE'], color=COLORS[:n])
    axes[1].set_xlabel('MAE'); axes[1].set_title('Mean Absolute Error', fontweight='bold')
    for bar, val in zip(bars, data['MAE']):
        axes[1].text(bar.get_width() * 1.02, bar.get_y() + bar.get_height()/2,
                     f'{val:.1f}', va='center', fontsize=10)
    bars = axes[2].barh(data['Target'], data['RMSE'], color=COLORS[:n])
    axes[2].set_xlabel('RMSE'); axes[2].set_title('RMSE', fontweight='bold')
    for bar, val in zip(bars, data['RMSE']):
        axes[2].text(bar.get_width() * 1.02, bar.get_y() + bar.get_height()/2,
                     f'{val:.1f}', va='center', fontsize=10)
    plt.suptitle('Multi-Output Regression — Performance Metrics (Ensemble)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'metrics_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ Saved metrics_comparison.png")


def plot_r2_evolution(results_dir):
    path = os.path.join(results_dir, 'training_history.json')
    if not os.path.exists(path):
        return
    with open(path, 'r') as f:
        history = json.load(f)
    if not history.get('val_metrics'):
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, name in enumerate(TARGET_NAMES):
        r2_vals = [em.get(name, {}).get('R2', np.nan) for em in history['val_metrics']]
        if not all(np.isnan(r2_vals)):
            ax.plot(range(1, len(r2_vals) + 1), r2_vals,
                    label=name, color=COLORS[idx], linewidth=2, alpha=0.85)
    ax.set_xlabel('Epoch'); ax.set_ylabel('R² Score')
    ax.set_title('R² Evolution During Training', fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3); ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'r2_evolution.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ Saved r2_evolution.png")


def evaluate():
    print("\n" + "█" * 60)
    print("  MPEA Multi-Output Regression — Ensemble Evaluation")
    print("█" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    models, feature_scaler, target_scalers, X_test_raw, Y_test_raw, metadata, device = \
        load_ensemble(base_dir)
    print(f"\n✓ Loaded {len(models)} ensemble models")
    print(f"✓ Test set: {len(X_test_raw)} samples")

    Y_pred = ensemble_predict(models, X_test_raw, feature_scaler, target_scalers, device)

    print(f"\n{'='*65}")
    print(f"  TEST SET METRICS (Original Scale, {len(models)}-Model Ensemble)")
    print(f"{'='*65}")
    print(f"\n{'Target':<25} {'R²':>8} {'MAE':>10} {'RMSE':>10} {'N':>6}")
    print("─" * 65)
    r2_list = []
    for i, name in enumerate(TARGET_NAMES):
        valid = ~np.isnan(Y_test_raw[:, i])
        if valid.sum() < 5:
            print(f"{name:<25} {'N/A':>8}")
            continue
        true_i, pred_i = Y_test_raw[valid, i], Y_pred[valid, i]
        r2 = r2_score(true_i, pred_i)
        mae = mean_absolute_error(true_i, pred_i)
        rmse = np.sqrt(mean_squared_error(true_i, pred_i))
        r2_list.append(r2)
        print(f"{name:<25} {r2:>8.4f} {mae:>10.2f} {rmse:>10.2f} {valid.sum():>6}")
    print(f"\n{'OVERALL AVG R²':<25} {np.mean(r2_list):>8.4f}")

    print(f"\nGenerating plots...")
    plot_parity(Y_test_raw, Y_pred, results_dir)
    plot_training_curves(results_dir)
    plot_metrics_bar(Y_test_raw, Y_pred, results_dir)
    plot_r2_evolution(results_dir)
    print(f"\n✓ All plots saved to: {results_dir}/")


if __name__ == '__main__':
    evaluate()
