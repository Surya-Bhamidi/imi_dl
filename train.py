"""
Enhanced Training Pipeline with Multi-Seed Ensemble
=====================================================
Trains multiple models with different seeds and aggregates predictions
for robust, high-accuracy results.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

from data_preprocessing import load_and_preprocess, TARGET_NAMES
from model import MultiOutputRegressor, MaskedMSELoss, MaskedHuberLoss, count_parameters

import joblib

# ─── Configuration ──────────────────────────────────────────────────────────
CONFIG = {
    'epochs': 600,
    'learning_rate': 5e-4,
    'weight_decay': 1e-5,
    'dropout': 0.10,
    'patience': 50,
    'batch_size': 32,
    'loss_type': 'huber',
    'huber_delta': 0.5,
    'n_ensemble': 5,          # Number of ensemble models
    'ensemble_seeds': [42, 123, 456, 789, 2024],
}


def compute_per_target_metrics(predictions, targets, mask):
    """R² and MAE for each target."""
    metrics = {}
    for i, name in enumerate(TARGET_NAMES):
        valid = mask[:, i] > 0.5
        if valid.sum() == 0:
            continue
        pred_i = predictions[valid, i]
        true_i = targets[valid, i]
        mae = np.mean(np.abs(pred_i - true_i))
        ss_res = np.sum((pred_i - true_i) ** 2)
        ss_tot = np.sum((true_i - np.mean(true_i)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        metrics[name] = {'R2': r2, 'MAE': mae}
    return metrics


def train_one_epoch(model, loader, optimizer, loss_fn, device, target_weights=None):
    model.train()
    total_loss = 0.0
    n = 0
    for features, targets, mask, weights in loader:
        features, targets, mask = features.to(device), targets.to(device), mask.to(device)
        
        optimizer.zero_grad()
        pred = model(features)
        
        loss = loss_fn(pred, targets, mask, target_weights=target_weights)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def validate(model, loader, loss_fn, device, target_weights=None):
    model.eval()
    total_loss = 0.0
    n = 0
    all_p, all_t, all_m = [], [], []
    with torch.no_grad():
        for features, targets, mask, weights in loader:
            features, targets, mask = features.to(device), targets.to(device), mask.to(device)
            pred = model(features)
            loss = loss_fn(pred, targets, mask, target_weights=target_weights)
            total_loss += loss.item()
            n += 1
            all_p.append(pred.cpu().numpy())
            all_t.append(targets.cpu().numpy())
            all_m.append(mask.cpu().numpy())
    avg_loss = total_loss / max(n, 1)
    all_p = np.concatenate(all_p)
    all_t = np.concatenate(all_t)
    all_m = np.concatenate(all_m)
    metrics = compute_per_target_metrics(all_p, all_t, all_m)
    return avg_loss, metrics


def train_single_model(seed, train_loader, val_loader, n_features, n_targets,
                        save_dir, device, model_idx):
    """Train a single model with a specific seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MultiOutputRegressor(
        n_features=n_features, n_targets=n_targets,
        dropout=CONFIG['dropout']
    ).to(device)

    loss_fn = MaskedHuberLoss(delta=CONFIG['huber_delta']) if CONFIG['loss_type'] == 'huber' else MaskedMSELoss()

    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'],
                      weight_decay=CONFIG['weight_decay'])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)

    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'val_metrics': [], 'lr': []}

    print(f"\n{'─'*50}")
    print(f"  Ensemble Model {model_idx+1}/{CONFIG['n_ensemble']} (seed={seed})")
    print(f"{'─'*50}")
    print(f"{'Epoch':>6} {'Train':>10} {'Val':>10} {'Best R²':>10}")

    for epoch in range(1, CONFIG['epochs'] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_metrics = validate(model, val_loader, loss_fn, device)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append({k: v for k, v in val_metrics.items()})
        history['lr'].append(lr)

        r2s = [m['R2'] for m in val_metrics.values()]
        best_r2 = max(r2s) if r2s else 0
        avg_r2 = np.mean(r2s) if r2s else 0

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6} {train_loss:>10.6f} {val_loss:>10.6f} {avg_r2:>10.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
            }, os.path.join(save_dir, f'model_{model_idx}.pth'))
        else:
            patience_counter += 1

        if patience_counter >= CONFIG['patience']:
            print(f"  ⏹ Early stop at epoch {epoch}")
            break

    # Load best weights
    ckpt = torch.load(os.path.join(save_dir, f'model_{model_idx}.pth'), weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  ✓ Best epoch: {best_epoch}, Val loss: {best_val_loss:.6f}")

    return model, history


def ensemble_predict(models, loader, device):
    """Average predictions from multiple models."""
    all_preds = []
    all_targets = []
    all_masks = []

    for features, targets, mask, weights in loader:
        features = features.to(device)
        batch_preds = []
        for model in models:
            model.eval()
            with torch.no_grad():
                pred = model(features).cpu().numpy()
            batch_preds.append(pred)

        # Average predictions across ensemble
        avg_pred = np.mean(batch_preds, axis=0)
        all_preds.append(avg_pred)
        all_targets.append(targets.numpy())
        all_masks.append(mask.numpy())

    return (np.concatenate(all_preds),
            np.concatenate(all_targets),
            np.concatenate(all_masks))


def train():
    """Main training function with ensemble."""
    print("\n" + "█" * 60)
    print("  MPEA Multi-Output Regression — Ensemble Training")
    print("█" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, 'saved_models')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n⚡ Device: {device}")

    csv_path = os.path.join(base_dir, 'MPEA_dataset_clean.csv')
    (train_loader, val_loader, test_loader,
     feature_scaler, target_scalers,
     feature_names, category_mappings, metadata,
     X_test_raw, Y_test_raw) = load_and_preprocess(csv_path)

    # Save artifacts
    joblib.dump(feature_scaler, os.path.join(save_dir, 'feature_scaler.pkl'))
    joblib.dump(target_scalers, os.path.join(save_dir, 'target_scalers.pkl'))
    with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(save_dir, 'category_mappings.json'), 'w') as f:
        json.dump(category_mappings, f, indent=2)
    np.save(os.path.join(save_dir, 'X_test_raw.npy'), X_test_raw.astype(np.float64))
    np.save(os.path.join(save_dir, 'Y_test_raw.npy'), Y_test_raw.astype(np.float64))

    n_features = metadata['n_features']
    n_targets = metadata['n_targets']

    print(f"\n🏗  Model Info:")
    print(f"   • Features:    {n_features}")
    print(f"   • Targets:     {n_targets}")
    print(f"   • Ensemble:    {CONFIG['n_ensemble']} models")
    test_model = MultiOutputRegressor(n_features, n_targets, CONFIG['dropout'])
    print(f"   • Params/model: {count_parameters(test_model):,}")
    del test_model

    # ─── Train ensemble ─────────────────────────────────────────────────
    start_time = time.time()
    models = []
    all_histories = []

    for i, seed in enumerate(CONFIG['ensemble_seeds'][:CONFIG['n_ensemble']]):
        model, history = train_single_model(
            seed, train_loader, val_loader,
            n_features, n_targets, save_dir, device, i
        )
        models.append(model)
        all_histories.append(history)

    elapsed = time.time() - start_time
    print(f"\n✓ All {CONFIG['n_ensemble']} models trained in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ─── Ensemble evaluation on test set ─────────────────────────────────
    loss_fn = MaskedHuberLoss(delta=CONFIG['huber_delta'])

    # Single best model metrics
    single_best_r2 = -float('inf')
    single_best_idx = 0
    for i, model in enumerate(models):
        _, metrics_i = validate(model, test_loader, loss_fn, device)
        avg_r2_i = np.mean([m['R2'] for m in metrics_i.values()])
        if avg_r2_i > single_best_r2:
            single_best_r2 = avg_r2_i
            single_best_idx = i

    # Save best single model as best_model.pth for the web app
    best_ckpt = torch.load(os.path.join(save_dir, f'model_{single_best_idx}.pth'), weights_only=False)
    torch.save(best_ckpt, os.path.join(save_dir, 'best_model.pth'))

    # Ensemble predictions
    ens_preds, ens_targets, ens_masks = ensemble_predict(models, test_loader, device)
    test_metrics = compute_per_target_metrics(ens_preds, ens_targets, ens_masks)

    print(f"\n{'='*60}")
    print(f"  ENSEMBLE TEST RESULTS ({CONFIG['n_ensemble']} models)")
    print(f"{'='*60}")
    print(f"\n{'Target':<25} {'R²':>8} {'MAE':>10}")
    print("─" * 45)
    r2_list = []
    for name, m in test_metrics.items():
        print(f"{name:<25} {m['R2']:>8.4f} {m['MAE']:>10.4f}")
        r2_list.append(m['R2'])
    overall_r2 = np.mean(r2_list)
    print(f"\n{'OVERALL AVERAGE R²':<25} {overall_r2:>8.4f}")

    # Save history
    serializable = {
        'train_loss': [float(x) for x in all_histories[0]['train_loss']],
        'val_loss': [float(x) for x in all_histories[0]['val_loss']],
        'lr': [float(x) for x in all_histories[0]['lr']],
        'val_metrics': [],
        'test_metrics': {},
        'config': CONFIG,
        'n_ensemble': CONFIG['n_ensemble'],
    }
    for vm in all_histories[0]['val_metrics']:
        epoch_m = {}
        for k, v in vm.items():
            epoch_m[k] = {mk: float(mv) for mk, mv in v.items()}
        serializable['val_metrics'].append(epoch_m)
    for k, v in test_metrics.items():
        serializable['test_metrics'][k] = {mk: float(mv) for mk, mv in v.items()}

    with open(os.path.join(results_dir, 'training_history.json'), 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"\n✓ Models saved to: {save_dir}/")
    print(f"✓ History saved to: {results_dir}/training_history.json")

    return models, all_histories, test_metrics


if __name__ == '__main__':
    train()
