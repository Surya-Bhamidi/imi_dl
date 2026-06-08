"""
A/B Feature Comparison: Baseline vs Enhanced Physical Parameters
================================================================
Quick training (100 epochs, 1 model) to measure impact of adding
new elemental properties before committing to full ensemble training.
"""

import os, sys, time, json, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from data_preprocessing import (
    parse_formula, compute_domain_features, extract_composition_features,
    encode_categoricals, prepare_numeric_features, prepare_targets,
    remove_outliers, MPEADataset, ALL_ELEMENTS, TARGET_NAMES,
    ATOMIC_RADIUS, ELECTRONEGATIVITY, VEC_VALUES, ATOMIC_MASS,
    MELTING_POINT, BULK_MODULUS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS,
    TARGET_COLUMNS, R_GAS
)
from torch.utils.data import DataLoader
from model import MultiOutputRegressor, MaskedHuberLoss

# ─── NEW Elemental Property Tables ──────────────────────────────────────────

SHEAR_MODULUS = {  # GPa
    'Al': 26, 'B': 200, 'C': 12, 'Co': 75, 'Cr': 115, 'Cu': 48,
    'Fe': 82, 'Hf': 30, 'Li': 4.2, 'Mg': 17, 'Mn': 80, 'Mo': 120,
    'Nb': 38, 'Nd': 16, 'Ni': 76, 'O': 0.5, 'Sc': 29, 'Si': 65,
    'Sn': 18, 'Ta': 69, 'Ti': 44, 'V': 47, 'W': 161, 'Y': 26,
    'Zn': 43, 'Zr': 33,
}

YOUNGS_MODULUS_ELEM = {  # GPa
    'Al': 70, 'B': 400, 'C': 33, 'Co': 209, 'Cr': 279, 'Cu': 130,
    'Fe': 211, 'Hf': 78, 'Li': 5, 'Mg': 45, 'Mn': 198, 'Mo': 329,
    'Nb': 105, 'Nd': 41, 'Ni': 200, 'O': 1, 'Sc': 74, 'Si': 130,
    'Sn': 50, 'Ta': 186, 'Ti': 116, 'V': 128, 'W': 411, 'Y': 64,
    'Zn': 108, 'Zr': 68,
}

POISSONS_RATIO = {
    'Al': 0.35, 'B': 0.17, 'C': 0.20, 'Co': 0.31, 'Cr': 0.21, 'Cu': 0.34,
    'Fe': 0.29, 'Hf': 0.37, 'Li': 0.36, 'Mg': 0.29, 'Mn': 0.24, 'Mo': 0.31,
    'Nb': 0.40, 'Nd': 0.28, 'Ni': 0.31, 'O': 0.30, 'Sc': 0.28, 'Si': 0.22,
    'Sn': 0.36, 'Ta': 0.34, 'Ti': 0.32, 'V': 0.37, 'W': 0.28, 'Y': 0.24,
    'Zn': 0.25, 'Zr': 0.34,
}

ELEMENTAL_DENSITY = {  # g/cm³
    'Al': 2.70, 'B': 2.34, 'C': 2.27, 'Co': 8.90, 'Cr': 7.19, 'Cu': 8.96,
    'Fe': 7.87, 'Hf': 13.31, 'Li': 0.53, 'Mg': 1.74, 'Mn': 7.47, 'Mo': 10.28,
    'Nb': 8.57, 'Nd': 7.01, 'Ni': 8.91, 'O': 1.43, 'Sc': 2.99, 'Si': 2.33,
    'Sn': 7.31, 'Ta': 16.69, 'Ti': 4.51, 'V': 6.11, 'W': 19.25, 'Y': 4.47,
    'Zn': 7.13, 'Zr': 6.52,
}

THERMAL_CONDUCTIVITY = {  # W/(m·K)
    'Al': 237, 'B': 27, 'C': 140, 'Co': 100, 'Cr': 94, 'Cu': 401,
    'Fe': 80, 'Hf': 23, 'Li': 85, 'Mg': 156, 'Mn': 7.8, 'Mo': 139,
    'Nb': 54, 'Nd': 17, 'Ni': 91, 'O': 0.03, 'Sc': 16, 'Si': 149,
    'Sn': 67, 'Ta': 57, 'Ti': 22, 'V': 31, 'W': 174, 'Y': 17,
    'Zn': 116, 'Zr': 23,
}

D_ELECTRONS = {
    'Al': 0, 'B': 0, 'C': 0, 'Co': 7, 'Cr': 5, 'Cu': 10,
    'Fe': 6, 'Hf': 2, 'Li': 0, 'Mg': 0, 'Mn': 5, 'Mo': 5,
    'Nb': 4, 'Nd': 1, 'Ni': 8, 'O': 0, 'Sc': 1, 'Si': 0,
    'Sn': 0, 'Ta': 3, 'Ti': 2, 'V': 3, 'W': 4, 'Y': 1,
    'Zn': 10, 'Zr': 2,
}

IONIZATION_ENERGY = {  # eV (first)
    'Al': 5.99, 'B': 8.30, 'C': 11.26, 'Co': 7.88, 'Cr': 6.77, 'Cu': 7.73,
    'Fe': 7.90, 'Hf': 6.83, 'Li': 5.39, 'Mg': 7.65, 'Mn': 7.43, 'Mo': 7.09,
    'Nb': 6.76, 'Nd': 5.53, 'Ni': 7.64, 'O': 13.62, 'Sc': 6.56, 'Si': 8.15,
    'Sn': 7.34, 'Ta': 7.55, 'Ti': 6.83, 'V': 6.75, 'W': 7.86, 'Y': 6.22,
    'Zn': 9.39, 'Zr': 6.63,
}


def compute_enhanced_domain_features(composition: dict) -> dict:
    """Compute ALL domain features: original + new physical parameters."""
    base = compute_domain_features(composition)
    if not composition:
        extra_keys = [
            'avg_shear_mod', 'avg_youngs_mod', 'avg_poisson', 'avg_elem_density',
            'avg_thermal_cond', 'avg_d_electrons', 'avg_ionization',
            'shear_mod_std', 'youngs_mod_std', 'density_std_elem',
            'delta_shear', 'Tm_std', 'lambda_param',
            'thermal_cond_std', 'VEC_squared', 'pugh_ratio',
        ]
        for k in extra_keys:
            base[k] = 0.0
        return base

    elements = list(composition.keys())
    fracs = np.array([composition[e] for e in elements])

    # New averages
    shear = np.array([SHEAR_MODULUS.get(e, 50) for e in elements])
    youngs = np.array([YOUNGS_MODULUS_ELEM.get(e, 100) for e in elements])
    poisson = np.array([POISSONS_RATIO.get(e, 0.3) for e in elements])
    dens = np.array([ELEMENTAL_DENSITY.get(e, 5) for e in elements])
    therm = np.array([THERMAL_CONDUCTIVITY.get(e, 50) for e in elements])
    d_elec = np.array([D_ELECTRONS.get(e, 0) for e in elements])
    ioniz = np.array([IONIZATION_ENERGY.get(e, 7) for e in elements])

    avg_shear = float(np.sum(fracs * shear))
    avg_youngs = float(np.sum(fracs * youngs))
    avg_poisson = float(np.sum(fracs * poisson))
    avg_dens = float(np.sum(fracs * dens))
    avg_therm = float(np.sum(fracs * therm))
    avg_d = float(np.sum(fracs * d_elec))
    avg_ion = float(np.sum(fracs * ioniz))

    # Std devs (mismatch parameters)
    shear_std = float(np.sqrt(np.sum(fracs * (shear - avg_shear) ** 2)))
    youngs_std = float(np.sqrt(np.sum(fracs * (youngs - avg_youngs) ** 2)))
    dens_std = float(np.sqrt(np.sum(fracs * (dens - avg_dens) ** 2)))
    therm_std = float(np.sqrt(np.sum(fracs * (therm - avg_therm) ** 2)))

    # δ_shear (shear modulus mismatch, like δ for radius)
    delta_shear = float(np.sqrt(np.sum(fracs * (1 - shear / max(avg_shear, 1e-6)) ** 2))) * 100

    # Tm std
    tms = np.array([MELTING_POINT.get(e, 1500) for e in elements])
    avg_tm = float(np.sum(fracs * tms))
    tm_std = float(np.sqrt(np.sum(fracs * (tms - avg_tm) ** 2)))

    # λ = ΔSmix / δ²  (solid-solution formation parameter)
    delta = base.get('delta', 1e-6)
    delta_smix = base.get('DeltaSmix', 0)
    lambda_param = delta_smix / max(delta ** 2, 1e-6)

    # VEC²
    vec_sq = base.get('VEC', 0) ** 2

    # Pugh ratio (bulk/shear) — ductility indicator
    avg_bulk = base.get('avg_modulus', 100)
    pugh = avg_bulk / max(avg_shear, 1e-6)

    base.update({
        'avg_shear_mod': avg_shear,
        'avg_youngs_mod': avg_youngs,
        'avg_poisson': avg_poisson,
        'avg_elem_density': avg_dens,
        'avg_thermal_cond': avg_therm,
        'avg_d_electrons': avg_d,
        'avg_ionization': avg_ion,
        'shear_mod_std': shear_std,
        'youngs_mod_std': youngs_std,
        'density_std_elem': dens_std,
        'delta_shear': delta_shear,
        'Tm_std': tm_std,
        'lambda_param': min(lambda_param, 1000),
        'thermal_cond_std': therm_std,
        'VEC_squared': vec_sq,
        'pugh_ratio': min(pugh, 100),
    })
    return base


def build_features(df, use_enhanced=False, use_extra_csv_cols=False):
    """Build feature matrix. If use_enhanced, adds new physical parameters."""
    compositions = df['FORMULA'].apply(parse_formula)

    # Elemental fractions
    comp_df = pd.DataFrame(compositions.tolist(), index=df.index)
    for elem in ALL_ELEMENTS:
        if elem not in comp_df.columns:
            comp_df[elem] = 0.0
    comp_df = comp_df[ALL_ELEMENTS].fillna(0.0)

    # Domain features
    if use_enhanced:
        domain_feats = compositions.apply(compute_enhanced_domain_features)
    else:
        domain_feats = compositions.apply(compute_domain_features)
    domain_df = pd.DataFrame(domain_feats.tolist(), index=df.index)

    # Categorical + numeric
    cat_features, _ = encode_categoricals(df)
    num_features = prepare_numeric_features(df)

    parts = [comp_df, domain_df, cat_features, num_features]

    # Extra CSV columns not currently used
    if use_extra_csv_cols:
        extra_cols = [
            'PROPERTY: O content (wppm)',
            'PROPERTY: N content (wppm)',
            'PROPERTY: C content (wppm)',
            'PROPERTY: Elongation plastic (%)',
            'PROPERTY: Exp. Young modulus (GPa)',
        ]
        extra_df = pd.DataFrame(index=df.index)
        for col in extra_cols:
            if col in df.columns:
                short = col.split(': ')[-1][:20]
                extra_df[short] = pd.to_numeric(df[col], errors='coerce')
        extra_df = extra_df.fillna(extra_df.median()).fillna(0.0)
        parts.append(extra_df)

        # Type of test (categorical)
        if 'PROPERTY: Type of test' in df.columns:
            test_type = df['PROPERTY: Type of test'].fillna('Unknown').astype(str)
            test_dummies = pd.get_dummies(test_type, prefix='TestType')
            parts.append(test_dummies)

    X = pd.concat(parts, axis=1)
    return X


def quick_train_eval(X_arr, Y_arr, label="", epochs=100, seed=42):
    """Train a single model quickly and return test R² scores."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    W_arr = np.ones(len(X_arr))
    X_arr, Y_arr, W_arr = remove_outliers(X_arr, Y_arr, W_arr, factor=3.0)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_arr, Y_arr, test_size=0.2, random_state=seed)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=0.125, random_state=seed)

    # Scale features
    f_scaler = RobustScaler()
    X_tr = f_scaler.fit_transform(X_train)
    X_va = f_scaler.transform(X_val)
    X_te = f_scaler.transform(X_test)

    # Scale targets
    t_scalers = []
    Y_tr, Y_va, Y_te = Y_train.copy(), Y_val.copy(), Y_test.copy()
    for i in range(Y_train.shape[1]):
        sc = StandardScaler()
        v = ~np.isnan(Y_train[:, i])
        if v.sum() > 0:
            sc.fit(Y_train[v, i].reshape(-1, 1))
            Y_tr[v, i] = sc.transform(Y_train[v, i].reshape(-1, 1)).ravel()
            vv = ~np.isnan(Y_val[:, i])
            if vv.sum() > 0:
                Y_va[vv, i] = sc.transform(Y_val[vv, i].reshape(-1, 1)).ravel()
            vt = ~np.isnan(Y_test[:, i])
            if vt.sum() > 0:
                Y_te[vt, i] = sc.transform(Y_test[vt, i].reshape(-1, 1)).ravel()
        t_scalers.append(sc)

    train_ds = MPEADataset(X_tr, Y_tr)
    val_ds = MPEADataset(X_va, Y_va)
    train_ld = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=32)

    device = torch.device('cpu')
    n_feat = X_tr.shape[1]
    model = MultiOutputRegressor(n_features=n_feat, n_targets=5, dropout=0.12).to(device)
    loss_fn = MaskedHuberLoss(delta=0.5)
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)

    best_val = float('inf')
    patience = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for feats, tgts, mask, wts in train_ld:
            optimizer.zero_grad()
            pred = model(feats)
            loss = loss_fn(pred, tgts, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        vl = 0; n = 0
        with torch.no_grad():
            for feats, tgts, mask, wts in val_ld:
                loss = loss_fn(model(feats), tgts, mask)
                vl += loss.item(); n += 1
        vl /= max(n, 1)

        if vl < best_val:
            best_val = vl
            patience = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience += 1
        if patience >= 30:
            break

    model.load_state_dict(best_state)
    model.eval()

    # Evaluate on test set (original scale)
    X_te_t = torch.FloatTensor(X_te)
    with torch.no_grad():
        preds_scaled = model(X_te_t).numpy()

    preds_orig = np.zeros_like(preds_scaled)
    for i, sc in enumerate(t_scalers):
        if hasattr(sc, 'mean_') and sc.mean_ is not None:
            preds_orig[:, i] = sc.inverse_transform(preds_scaled[:, i].reshape(-1, 1)).ravel()
        else:
            preds_orig[:, i] = preds_scaled[:, i]

    results = {}
    for i, name in enumerate(TARGET_NAMES):
        valid = ~np.isnan(Y_test[:, i])
        if valid.sum() < 5:
            continue
        r2 = r2_score(Y_test[valid, i], preds_orig[valid, i])
        mae = mean_absolute_error(Y_test[valid, i], preds_orig[valid, i])
        results[name] = {'R2': round(r2, 4), 'MAE': round(mae, 2)}

    return results, n_feat


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'MPEA_dataset.csv')
    df = pd.read_csv(csv_path)

    Y = prepare_targets(df)
    valid_mask = Y.notna().any(axis=1)
    df_valid = df[valid_mask].reset_index(drop=True)
    Y_arr = Y[valid_mask].reset_index(drop=True).values.astype(np.float64)

    configs = [
        ("A: Baseline (current)", False, False),
        ("B: + New Elemental Props", True, False),
        ("C: + New Props + CSV Cols", True, True),
    ]

    all_results = {}
    for label, enhanced, extra_csv in configs:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        X = build_features(df_valid, use_enhanced=enhanced, use_extra_csv_cols=extra_csv)
        X_arr = X.values.astype(np.float64)
        print(f"  Features: {X_arr.shape[1]}")

        t0 = time.time()
        results, n_feat = quick_train_eval(X_arr, Y_arr, label=label, epochs=100)
        elapsed = time.time() - t0
        print(f"  Trained in {elapsed:.1f}s")

        print(f"\n  {'Target':<25} {'R²':>8} {'MAE':>10}")
        print(f"  {'─'*45}")
        r2s = []
        for name, m in results.items():
            print(f"  {name:<25} {m['R2']:>8.4f} {m['MAE']:>10.2f}")
            r2s.append(m['R2'])
        avg = np.mean(r2s) if r2s else 0
        print(f"\n  {'AVG R²':<25} {avg:>8.4f}")
        all_results[label] = {'metrics': results, 'n_features': n_feat, 'avg_r2': round(avg, 4)}

    # ─── Summary Comparison ─────────────────────────────────────────────
    print(f"\n\n{'█'*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'█'*60}")
    print(f"\n  {'Config':<30} {'Features':>8} {'Avg R²':>8} {'Δ vs Base':>10}")
    print(f"  {'─'*58}")
    base_r2 = list(all_results.values())[0]['avg_r2']
    for label, data in all_results.items():
        delta = data['avg_r2'] - base_r2
        sign = '+' if delta >= 0 else ''
        print(f"  {label:<30} {data['n_features']:>8} {data['avg_r2']:>8.4f} {sign}{delta:>9.4f}")

    # Per-target comparison
    print(f"\n  Per-Target R² Comparison:")
    print(f"  {'Target':<25}", end="")
    for label in all_results:
        short = label.split(':')[0]
        print(f" {short:>10}", end="")
    print(f" {'Best':>6}")
    print(f"  {'─'*65}")

    for name in TARGET_NAMES:
        print(f"  {name:<25}", end="")
        vals = []
        for label, data in all_results.items():
            r2 = data['metrics'].get(name, {}).get('R2', float('nan'))
            vals.append(r2)
            print(f" {r2:>10.4f}", end="")
        best_idx = np.nanargmax(vals)
        best_label = list(all_results.keys())[best_idx].split(':')[0]
        print(f" {best_label:>6}")

    # Save results
    out_path = os.path.join(base_dir, '_feature_comparison.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  ✓ Results saved to _feature_comparison.json")


if __name__ == '__main__':
    main()
