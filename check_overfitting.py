import numpy as np
import joblib
import pandas as pd
from data_preprocessing import load_and_preprocess, TARGET_NAMES
from sklearn.metrics import r2_score

def check_overfitting():
    csv_path = 'MPEA_dataset_clean.csv'
    # 1. Load data via the standard pipeline
    (train_loader, val_loader, test_loader,
     feature_scaler, target_scalers,
     feature_names, category_mappings, metadata,
     X_test_orig, Y_test_orig) = load_and_preprocess(csv_path)

    # 2. Extract arrays from datasets
    X_train_full = train_loader.dataset.features.numpy()
    Y_train_full = train_loader.dataset.targets.numpy()
    M_train_full = train_loader.dataset.mask.numpy()
    
    X_test_full = test_loader.dataset.features.numpy()
    Y_test_full = test_loader.dataset.targets.numpy()
    M_test_full = test_loader.dataset.mask.numpy()

    # 3. Load trained model
    models = joblib.load('saved_models/rf_models.pkl')
    
    overfit_data = {}
    
    for i, name in enumerate(TARGET_NAMES):
        model = models[i]
        
        # Test Scores
        valid_test = M_test_full[:, i] > 0.5
        y_test_pred = model.predict(X_test_full[valid_test])
        test_r2 = r2_score(Y_test_full[valid_test, i], y_test_pred)
        
        # Train Scores
        valid_train = M_train_full[:, i] > 0.5
        y_train_pred = model.predict(X_train_full[valid_train])
        train_r2 = r2_score(Y_train_full[valid_train, i], y_train_pred)
        
        overfit_data[name] = {
            "Train R2": round(train_r2, 4),
            "Test R2": round(test_r2, 4),
            "Gap": round(train_r2 - test_r2, 4)
        }
        
    import json
    print(json.dumps(overfit_data, indent=2))

if __name__ == "__main__":
    check_overfitting()
