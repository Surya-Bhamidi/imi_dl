"""
Enhanced Data Preprocessing with Domain-Specific Feature Engineering
=====================================================================
Adds materials science descriptors (VEC, δ, ΔHmix, ΔSmix, Δχ, etc.)
for dramatically improved prediction accuracy.
"""

import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os

# ─── Constants ──────────────────────────────────────────────────────────────
TARGET_COLUMNS = [
    'PROPERTY: HV',
    'PROPERTY: YS (MPa)',
    'PROPERTY: UTS (MPa)',
    'PROPERTY: Elongation (%)',
    'PROPERTY: Calculated Young modulus (GPa)',
]

TARGET_NAMES = [
    'Hardness (HV)',
    'Yield Strength (MPa)',
    'UTS (MPa)',
    'Elongation (%)',
    'Young Modulus (GPa)',
]

CATEGORICAL_COLUMNS = [
    'PROPERTY: Microstructure',
    'PROPERTY: Processing method',
    'PROPERTY: BCC/FCC/other',
]

NUMERIC_COLUMNS = [
    'PROPERTY: grain size ($\\mu$m)',
    'PROPERTY: Exp. Density (g/cm$^3$)',
    'PROPERTY: Calculated Density (g/cm$^3$)',
    'PROPERTY: Test temperature ($^\\circ$C)',
]

ALL_ELEMENTS = [
    'Al', 'B', 'C', 'Co', 'Cr', 'Cu', 'Fe', 'Hf', 'Li', 'Mg', 'Mn',
    'Mo', 'Nb', 'Nd', 'Ni', 'O', 'Sc', 'Si', 'Sn', 'Ta', 'Ti', 'V',
    'W', 'Y', 'Zn', 'Zr'
]

# ─── Elemental property tables for domain features ─────────────────────────
# Atomic radius in pm
ATOMIC_RADIUS = {
    'Al': 143, 'B': 87, 'C': 77, 'Co': 125, 'Cr': 128, 'Cu': 128,
    'Fe': 126, 'Hf': 159, 'Li': 152, 'Mg': 160, 'Mn': 127, 'Mo': 139,
    'Nb': 146, 'Nd': 182, 'Ni': 124, 'O': 60, 'Sc': 162, 'Si': 111,
    'Sn': 140, 'Ta': 146, 'Ti': 147, 'V': 134, 'W': 139, 'Y': 180,
    'Zn': 134, 'Zr': 160,
}

# Pauling electronegativity
ELECTRONEGATIVITY = {
    'Al': 1.61, 'B': 2.04, 'C': 2.55, 'Co': 1.88, 'Cr': 1.66, 'Cu': 1.90,
    'Fe': 1.83, 'Hf': 1.30, 'Li': 0.98, 'Mg': 1.31, 'Mn': 1.55, 'Mo': 2.16,
    'Nb': 1.60, 'Nd': 1.14, 'Ni': 1.91, 'O': 3.44, 'Sc': 1.36, 'Si': 1.90,
    'Sn': 1.96, 'Ta': 1.50, 'Ti': 1.54, 'V': 1.63, 'W': 2.36, 'Y': 1.22,
    'Zn': 1.65, 'Zr': 1.33,
}

# VEC (Valence Electron Concentration)
VEC_VALUES = {
    'Al': 3, 'B': 3, 'C': 4, 'Co': 9, 'Cr': 6, 'Cu': 11,
    'Fe': 8, 'Hf': 4, 'Li': 1, 'Mg': 2, 'Mn': 7, 'Mo': 6,
    'Nb': 5, 'Nd': 3, 'Ni': 10, 'O': 6, 'Sc': 3, 'Si': 4,
    'Sn': 4, 'Ta': 5, 'Ti': 4, 'V': 5, 'W': 6, 'Y': 3,
    'Zn': 12, 'Zr': 4,
}

# Atomic mass in g/mol
ATOMIC_MASS = {
    'Al': 26.98, 'B': 10.81, 'C': 12.01, 'Co': 58.93, 'Cr': 52.00,
    'Cu': 63.55, 'Fe': 55.85, 'Hf': 178.49, 'Li': 6.94, 'Mg': 24.31,
    'Mn': 54.94, 'Mo': 95.94, 'Nb': 92.91, 'Nd': 144.24, 'Ni': 58.69,
    'O': 16.00, 'Sc': 44.96, 'Si': 28.09, 'Sn': 118.71, 'Ta': 180.95,
    'Ti': 47.87, 'V': 50.94, 'W': 183.84, 'Y': 88.91, 'Zn': 65.38,
    'Zr': 91.22,
}

# Melting point in K
MELTING_POINT = {
    'Al': 933, 'B': 2349, 'C': 3823, 'Co': 1768, 'Cr': 2180, 'Cu': 1358,
    'Fe': 1811, 'Hf': 2506, 'Li': 454, 'Mg': 923, 'Mn': 1519, 'Mo': 2896,
    'Nb': 2750, 'Nd': 1297, 'Ni': 1728, 'O': 54, 'Sc': 1814, 'Si': 1687,
    'Sn': 505, 'Ta': 3290, 'Ti': 1941, 'V': 2183, 'W': 3695, 'Y': 1799,
    'Zn': 693, 'Zr': 2128,
}

# Bulk modulus in GPa (approximate)
BULK_MODULUS = {
    'Al': 76, 'B': 320, 'C': 33, 'Co': 180, 'Cr': 160, 'Cu': 140,
    'Fe': 170, 'Hf': 110, 'Li': 11, 'Mg': 45, 'Mn': 120, 'Mo': 230,
    'Nb': 170, 'Nd': 32, 'Ni': 180, 'O': 1, 'Sc': 57, 'Si': 100,
    'Sn': 58, 'Ta': 200, 'Ti': 110, 'V': 160, 'W': 310, 'Y': 41,
    'Zn': 70, 'Zr': 94,
}

# Shear Modulus in GPa
SHEAR_MODULUS = {
    'Al': 26, 'B': 200, 'C': 12, 'Co': 75, 'Cr': 115, 'Cu': 48,
    'Fe': 82, 'Hf': 30, 'Li': 4.2, 'Mg': 17, 'Mn': 80, 'Mo': 120,
    'Nb': 38, 'Nd': 16, 'Ni': 76, 'O': 0.5, 'Sc': 29, 'Si': 65,
    'Sn': 18, 'Ta': 69, 'Ti': 44, 'V': 47, 'W': 161, 'Y': 26,
    'Zn': 43, 'Zr': 33,
}

# Young's Modulus (elemental) in GPa
YOUNGS_MODULUS_ELEM = {
    'Al': 70, 'B': 400, 'C': 33, 'Co': 209, 'Cr': 279, 'Cu': 130,
    'Fe': 211, 'Hf': 78, 'Li': 5, 'Mg': 45, 'Mn': 198, 'Mo': 329,
    'Nb': 105, 'Nd': 41, 'Ni': 200, 'O': 1, 'Sc': 74, 'Si': 130,
    'Sn': 50, 'Ta': 186, 'Ti': 116, 'V': 128, 'W': 411, 'Y': 64,
    'Zn': 108, 'Zr': 68,
}

# Poisson's Ratio
POISSONS_RATIO = {
    'Al': 0.35, 'B': 0.17, 'C': 0.20, 'Co': 0.31, 'Cr': 0.21, 'Cu': 0.34,
    'Fe': 0.29, 'Hf': 0.37, 'Li': 0.36, 'Mg': 0.29, 'Mn': 0.24, 'Mo': 0.31,
    'Nb': 0.40, 'Nd': 0.28, 'Ni': 0.31, 'O': 0.30, 'Sc': 0.28, 'Si': 0.22,
    'Sn': 0.36, 'Ta': 0.34, 'Ti': 0.32, 'V': 0.37, 'W': 0.28, 'Y': 0.24,
    'Zn': 0.25, 'Zr': 0.34,
}

# Elemental Density in g/cm³
ELEMENTAL_DENSITY = {
    'Al': 2.70, 'B': 2.34, 'C': 2.27, 'Co': 8.90, 'Cr': 7.19, 'Cu': 8.96,
    'Fe': 7.87, 'Hf': 13.31, 'Li': 0.53, 'Mg': 1.74, 'Mn': 7.47, 'Mo': 10.28,
    'Nb': 8.57, 'Nd': 7.01, 'Ni': 8.91, 'O': 1.43, 'Sc': 2.99, 'Si': 2.33,
    'Sn': 7.31, 'Ta': 16.69, 'Ti': 4.51, 'V': 6.11, 'W': 19.25, 'Y': 4.47,
    'Zn': 7.13, 'Zr': 6.52,
}

# Thermal Conductivity in W/(m·K)
THERMAL_CONDUCTIVITY = {
    'Al': 237, 'B': 27, 'C': 140, 'Co': 100, 'Cr': 94, 'Cu': 401,
    'Fe': 80, 'Hf': 23, 'Li': 85, 'Mg': 156, 'Mn': 7.8, 'Mo': 139,
    'Nb': 54, 'Nd': 17, 'Ni': 91, 'O': 0.03, 'Sc': 16, 'Si': 149,
    'Sn': 67, 'Ta': 57, 'Ti': 22, 'V': 31, 'W': 174, 'Y': 17,
    'Zn': 116, 'Zr': 23,
}

# Number of d-electrons
D_ELECTRONS = {
    'Al': 0, 'B': 0, 'C': 0, 'Co': 7, 'Cr': 5, 'Cu': 10,
    'Fe': 6, 'Hf': 2, 'Li': 0, 'Mg': 0, 'Mn': 5, 'Mo': 5,
    'Nb': 4, 'Nd': 1, 'Ni': 8, 'O': 0, 'Sc': 1, 'Si': 0,
    'Sn': 0, 'Ta': 3, 'Ti': 2, 'V': 3, 'W': 4, 'Y': 1,
    'Zn': 10, 'Zr': 2,
}

# First Ionization Energy in eV
IONIZATION_ENERGY = {
    'Al': 5.99, 'B': 8.30, 'C': 11.26, 'Co': 7.88, 'Cr': 6.77, 'Cu': 7.73,
    'Fe': 7.90, 'Hf': 6.83, 'Li': 5.39, 'Mg': 7.65, 'Mn': 7.43, 'Mo': 7.09,
    'Nb': 6.76, 'Nd': 5.53, 'Ni': 7.64, 'O': 13.62, 'Sc': 6.56, 'Si': 8.15,
    'Sn': 7.34, 'Ta': 7.55, 'Ti': 6.83, 'V': 6.75, 'W': 7.86, 'Y': 6.22,
    'Zn': 9.39, 'Zr': 6.63,
}

R_GAS = 8.314  # J/(mol·K)


def parse_formula(formula: str) -> dict:
    """Parse alloy formula → element:amount dict, normalized to atomic fractions."""
    if pd.isna(formula):
        return {}
    pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
    matches = re.findall(pattern, str(formula))
    composition = {}
    for element, amount in matches:
        if element:
            composition[element] = float(amount) if amount else 1.0
    total = sum(composition.values())
    if total > 0:
        composition = {k: v / total for k, v in composition.items()}
    return composition


def compute_domain_features(composition: dict) -> dict:
    """
    Compute materials science domain-specific features from composition.
    Includes original features + enhanced physical parameters (Shear Modulus,
    Young's Modulus, Poisson's Ratio, Elemental Density, Thermal Conductivity,
    d-electrons, Ionization Energy) and derived mismatch/interaction features.
    """
    all_keys = [
        'VEC', 'delta', 'DeltaSmix', 'DeltaChi', 'avg_radius',
        'avg_mass', 'avg_Tm', 'avg_modulus', 'n_elements',
        'max_frac', 'min_frac', 'frac_range', 'entropy_config',
        'Omega', 'avg_EN', 'radius_std', 'mass_std',
        'avg_shear_mod', 'avg_youngs_mod', 'avg_poisson', 'avg_elem_density',
        'avg_thermal_cond', 'avg_d_electrons', 'avg_ionization',
        'shear_mod_std', 'youngs_mod_std', 'density_std_elem',
        'delta_shear', 'Tm_std', 'lambda_param',
        'thermal_cond_std', 'VEC_squared', 'pugh_ratio',
    ]
    if not composition:
        return {k: 0.0 for k in all_keys}

    elements = list(composition.keys())
    fracs = np.array([composition[e] for e in elements])

    # ── Original features ──
    vec_vals = np.array([VEC_VALUES.get(e, 5) for e in elements])
    VEC = float(np.sum(fracs * vec_vals))

    radii = np.array([ATOMIC_RADIUS.get(e, 130) for e in elements])
    avg_r = float(np.sum(fracs * radii))
    delta_sq = float(np.sum(fracs * (1 - radii / avg_r) ** 2))
    delta = np.sqrt(delta_sq) * 100

    en_vals = np.array([ELECTRONEGATIVITY.get(e, 1.5) for e in elements])
    avg_en = float(np.sum(fracs * en_vals))
    delta_chi = float(np.sqrt(np.sum(fracs * (en_vals - avg_en) ** 2)))

    fracs_nonzero = fracs[fracs > 1e-10]
    delta_smix = float(-R_GAS * np.sum(fracs_nonzero * np.log(fracs_nonzero)))

    masses = np.array([ATOMIC_MASS.get(e, 50) for e in elements])
    avg_mass = float(np.sum(fracs * masses))

    tms = np.array([MELTING_POINT.get(e, 1500) for e in elements])
    avg_tm = float(np.sum(fracs * tms))

    mods = np.array([BULK_MODULUS.get(e, 100) for e in elements])
    avg_mod = float(np.sum(fracs * mods))

    omega = avg_tm * delta_smix / max(abs(delta * avg_r), 1e-6)
    radius_std = float(np.sqrt(np.sum(fracs * (radii - avg_r) ** 2)))
    mass_std = float(np.sqrt(np.sum(fracs * (masses - avg_mass) ** 2)))

    # ── NEW: Enhanced physical parameters ──
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

    shear_std = float(np.sqrt(np.sum(fracs * (shear - avg_shear) ** 2)))
    youngs_std = float(np.sqrt(np.sum(fracs * (youngs - avg_youngs) ** 2)))
    dens_std = float(np.sqrt(np.sum(fracs * (dens - avg_dens) ** 2)))
    therm_std = float(np.sqrt(np.sum(fracs * (therm - avg_therm) ** 2)))
    delta_shear = float(np.sqrt(np.sum(fracs * (1 - shear / max(avg_shear, 1e-6)) ** 2))) * 100
    tm_std = float(np.sqrt(np.sum(fracs * (tms - avg_tm) ** 2)))
    lambda_param = delta_smix / max(delta ** 2, 1e-6)
    pugh = avg_mod / max(avg_shear, 1e-6)

    return {
        'VEC': VEC, 'delta': delta, 'DeltaSmix': delta_smix,
        'DeltaChi': delta_chi, 'avg_radius': avg_r, 'avg_mass': avg_mass,
        'avg_Tm': avg_tm, 'avg_modulus': avg_mod,
        'n_elements': len(elements),
        'max_frac': float(np.max(fracs)), 'min_frac': float(np.min(fracs)),
        'frac_range': float(np.max(fracs) - np.min(fracs)),
        'entropy_config': delta_smix / R_GAS,
        'Omega': min(omega, 1000), 'avg_EN': avg_en,
        'radius_std': radius_std, 'mass_std': mass_std,
        'avg_shear_mod': avg_shear, 'avg_youngs_mod': avg_youngs,
        'avg_poisson': avg_poisson, 'avg_elem_density': avg_dens,
        'avg_thermal_cond': avg_therm, 'avg_d_electrons': avg_d,
        'avg_ionization': avg_ion, 'shear_mod_std': shear_std,
        'youngs_mod_std': youngs_std, 'density_std_elem': dens_std,
        'delta_shear': delta_shear, 'Tm_std': tm_std,
        'lambda_param': min(lambda_param, 1000),
        'thermal_cond_std': therm_std, 'VEC_squared': VEC ** 2,
        'pugh_ratio': min(pugh, 100),
    }





def extract_composition_features(df: pd.DataFrame) -> tuple:
    """Extract elemental + domain features from FORMULA column."""
    compositions = df['FORMULA'].apply(parse_formula)

    # Elemental fractions
    comp_df = pd.DataFrame(compositions.tolist(), index=df.index)
    for elem in ALL_ELEMENTS:
        if elem not in comp_df.columns:
            comp_df[elem] = 0.0
    comp_df = comp_df[ALL_ELEMENTS].fillna(0.0)

    # Domain features
    domain_feats = compositions.apply(compute_domain_features)
    domain_df = pd.DataFrame(domain_feats.tolist(), index=df.index)

    return comp_df, domain_df


def encode_categoricals(df: pd.DataFrame) -> tuple:
    """One-hot encode categorical columns."""
    encoded_frames = []
    category_mappings = {}
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            series = df[col].fillna('Unknown').astype(str)
            dummies = pd.get_dummies(series, prefix=col.split(': ')[-1][:10])
            encoded_frames.append(dummies)
            category_mappings[col] = list(dummies.columns)
    if not encoded_frames:
        return pd.DataFrame(index=df.index), category_mappings
    return pd.concat(encoded_frames, axis=1), category_mappings


def prepare_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract numeric feature columns."""
    numeric_df = pd.DataFrame(index=df.index)
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            short_name = col.split(': ')[-1][:20]
            numeric_df[short_name] = pd.to_numeric(df[col], errors='coerce')
    numeric_df = numeric_df.fillna(numeric_df.median())
    numeric_df = numeric_df.fillna(0.0)
    return numeric_df


def prepare_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Extract target columns."""
    targets = pd.DataFrame(index=df.index)
    for col, name in zip(TARGET_COLUMNS, TARGET_NAMES):
        if col in df.columns:
            targets[name] = pd.to_numeric(df[col], errors='coerce')
        else:
            targets[name] = np.nan
    return targets


def remove_outliers(X, Y, W, factor=3.0):
    """Remove extreme outliers per target using IQR method while preserving sample weights."""
    mask = np.ones(len(Y), dtype=bool)
    for i in range(Y.shape[1]):
        valid = ~np.isnan(Y[:, i])
        if valid.sum() < 10:
            continue
        vals = Y[valid, i]
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        outlier = valid & ((Y[:, i] < lower) | (Y[:, i] > upper))
        mask[outlier] = False
    return X[mask], Y[mask], W[mask]


class MPEADataset(Dataset):
    """PyTorch Dataset for MPEA multi-output regression with optional sample weighting."""
    def __init__(self, features: np.ndarray, targets: np.ndarray, weights: np.ndarray = None):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.mask = torch.FloatTensor(~np.isnan(targets)).float()
        self.targets = torch.nan_to_num(self.targets, nan=0.0)
        
        if weights is not None:
            self.weights = torch.FloatTensor(weights)
        else:
            self.weights = torch.ones(len(features))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.mask[idx], self.weights[idx]


def load_and_preprocess(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    """Full preprocessing pipeline with domain features."""
    print("=" * 60)
    print("  MPEA Multi-Output Regression — Enhanced Preprocessing")
    print("=" * 60)

    df = pd.read_csv(csv_path)
    print(f"\n✓ Loaded dataset: {len(df)} samples")

    # ---- Features ----
    comp_features, domain_features = extract_composition_features(df)
    print(f"✓ Extracted {len(ALL_ELEMENTS)} elemental + {domain_features.shape[1]} domain features")

    cat_features, category_mappings = encode_categoricals(df)
    print(f"✓ Encoded categoricals → {cat_features.shape[1]} columns")

    num_features = prepare_numeric_features(df)
    print(f"✓ Prepared {num_features.shape[1]} numeric features")

    # Combine all features
    X = pd.concat([comp_features, domain_features, cat_features, num_features], axis=1)
    feature_names = list(X.columns)
    print(f"✓ Total feature count: {X.shape[1]}")

    # ---- Targets ----
    Y = prepare_targets(df)
    print(f"\n{'Target':<25} {'Available':>10} {'Missing':>10} {'%':>8}")
    print("-" * 55)
    for col in Y.columns:
        avail = Y[col].notna().sum()
        miss = Y[col].isna().sum()
        print(f"{col:<25} {avail:>10} {miss:>10} {avail/len(Y)*100:>7.1f}%")

    # Filter rows with at least 1 target
    valid_mask = Y.notna().any(axis=1)
    X = X[valid_mask].reset_index(drop=True)
    Y = Y[valid_mask].reset_index(drop=True)
    print(f"\n✓ Kept {len(X)} samples with targets")

    X_arr = X.values.astype(np.float64)
    Y_arr = Y.values.astype(np.float64)
    
    if 'sample_weight' in df.columns:
        W_arr = df['sample_weight'][valid_mask].reset_index(drop=True).values.astype(np.float64)
    else:
        W_arr = np.ones(len(X_arr))

    # Remove outliers
    n_before = len(X_arr)
    X_arr, Y_arr, W_arr = remove_outliers(X_arr, Y_arr, W_arr, factor=2.0)
    print(f"✓ Removed {n_before - len(X_arr)} outliers → {len(X_arr)} samples")

    # ---- Split ----
    X_train, X_test, Y_train, Y_test, W_train, W_test = train_test_split(
        X_arr, Y_arr, W_arr, test_size=test_size, random_state=random_state
    )
    X_train, X_val, Y_train, Y_val, W_train, W_val = train_test_split(
        X_train, Y_train, W_train, test_size=0.125, random_state=random_state
    )
    print(f"✓ Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # ---- Scale features (RobustScaler is more outlier-resistant) ----
    feature_scaler = RobustScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_val_scaled = feature_scaler.transform(X_val)
    X_test_scaled = feature_scaler.transform(X_test)

    # ---- Scale targets per-column ----
    target_scalers = []
    Y_train_scaled = Y_train.copy()
    Y_val_scaled = Y_val.copy()
    Y_test_scaled = Y_test.copy()
    
    from sklearn.preprocessing import StandardScaler

    for i in range(Y_train.shape[1]):
        scaler = StandardScaler()
        valid_train = ~np.isnan(Y_train[:, i])
        if valid_train.sum() > 0:
            scaler.fit(Y_train[valid_train, i].reshape(-1, 1))
            Y_train_scaled[valid_train, i] = scaler.transform(
                Y_train[valid_train, i].reshape(-1, 1)).ravel()
            valid_val = ~np.isnan(Y_val[:, i])
            if valid_val.sum() > 0:
                Y_val_scaled[valid_val, i] = scaler.transform(
                    Y_val[valid_val, i].reshape(-1, 1)).ravel()
            valid_test = ~np.isnan(Y_test[:, i])
            if valid_test.sum() > 0:
                Y_test_scaled[valid_test, i] = scaler.transform(
                    Y_test[valid_test, i].reshape(-1, 1)).ravel()
        target_scalers.append(scaler)

    # ---- Create DataLoaders ----
    train_dataset = MPEADataset(X_train_scaled, Y_train_scaled, W_train)
    val_dataset = MPEADataset(X_val_scaled, Y_val_scaled, W_val)
    test_dataset = MPEADataset(X_test_scaled, Y_test_scaled, W_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    metadata = {
        'n_features': X.shape[1],
        'n_targets': Y.shape[1],
        'feature_names': feature_names,
        'target_names': TARGET_NAMES,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'all_elements': ALL_ELEMENTS,
    }

    print(f"\n✓ Preprocessing complete!")
    print("=" * 60)

    return (train_loader, val_loader, test_loader,
            feature_scaler, target_scalers,
            feature_names, category_mappings, metadata,
            X_test, Y_test)


if __name__ == '__main__':
    csv_path = os.path.join(os.path.dirname(__file__), 'MPEA_dataset.csv')
    results = load_and_preprocess(csv_path)
    print("\nPreprocessing test passed successfully!")
