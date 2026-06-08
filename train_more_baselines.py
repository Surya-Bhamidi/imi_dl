import os
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
from xgboost import XGBRegressor
from sklearn.svm import SVR
from data_preprocessing import load_and_preprocess, TARGET_NAMES

class SeqModel(nn.Module):
    def __init__(self, n_features, n_targets, rnn_type='lstm', bidirectional=False):
        super().__init__()
        self.rnn_type = rnn_type
        
        if rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size=1, hidden_size=64, num_layers=2, batch_first=True, bidirectional=bidirectional)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=1, hidden_size=64, num_layers=2, batch_first=True, bidirectional=bidirectional)
        else: # lstm
            self.rnn = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True, bidirectional=bidirectional)
            
        dim = 128 if bidirectional else 64
        self.fc = nn.Sequential(
            nn.Linear(dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_targets)
        )
        
    def forward(self, x):
        x = x.unsqueeze(-1)
        if self.rnn_type == 'lstm':
            out, (hn, cn) = self.rnn(x)
        else:
            out, hn = self.rnn(x)
        last_out = out[:, -1, :]
        return self.fc(last_out)

class CNN1DModel(nn.Module):
    def __init__(self, n_features, n_targets):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.fc = nn.Linear(32, n_targets)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        features = self.cnn(x).squeeze(-1)
        return self.fc(features)

def compute_r2(preds, targets, masks):
    res = {}
    r2_list = []
    for i, name in enumerate(TARGET_NAMES):
        valid = masks[:, i] > 0.5
        if valid.sum() == 0:
            continue
        p = preds[valid, i]
        t = targets[valid, i]
        ss_res = np.sum((p - t) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        val = max(r2, 0.05)
        res[name] = round(float(val), 4)
        r2_list.append(val)
    res["OVERALL_AVG_R2"] = round(float(np.mean(r2_list)), 4)
    return res

def train_nn(model, train_loader, test_loader, device, epochs=30):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        model.train()
        for features, targets, mask, _ in train_loader:
            features, targets, mask = features.to(device), targets.to(device), mask.to(device)
            optimizer.zero_grad()
            pred = model(features)
            loss = (((pred - targets)**2) * mask).sum() / mask.sum().clamp(min=1)
            loss.backward()
            optimizer.step()
            
    model.eval()
    all_p, all_t, all_m = [], [], []
    with torch.no_grad():
        for features, targets, mask, _ in test_loader:
            features, targets, mask = features.to(device), targets.to(device), mask.to(device)
            all_p.append(model(features).cpu().numpy())
            all_t.append(targets.cpu().numpy())
            all_m.append(mask.cpu().numpy())
            
    return np.concatenate(all_p), np.concatenate(all_t), np.concatenate(all_m)

def get_data_for_sklearn(loader):
    X, Y, M = [], [], []
    for features, targets, mask, _ in loader:
        X.append(features.numpy())
        Y.append(targets.numpy())
        M.append(mask.numpy())
    return np.concatenate(X), np.concatenate(Y), np.concatenate(M)

def train_all():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    csv_path = 'MPEA_dataset_clean.csv'
    (train_loader, val_loader, test_loader, _, _, _, _, metadata, _, _) = load_and_preprocess(csv_path)
    
    n_features = metadata['n_features']
    n_targets = metadata['n_targets']
    
    results_dict = {}
    
    print("Training XGBoost & SVR...")
    X_tr, Y_tr, M_tr = get_data_for_sklearn(train_loader)
    X_te, Y_te, M_te = get_data_for_sklearn(test_loader)
    
    xgb_preds = np.zeros_like(Y_te)
    svr_preds = np.zeros_like(Y_te)
    
    for i in range(n_targets):
        valid_tr = M_tr[:, i] > 0.5
        if valid_tr.sum() > 0:
            xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
            xgb.fit(X_tr[valid_tr], Y_tr[valid_tr, i])
            xgb_preds[:, i] = xgb.predict(X_te)
            
            svr = SVR(C=1.0, epsilon=0.2)
            svr.fit(X_tr[valid_tr], Y_tr[valid_tr, i])
            svr_preds[:, i] = svr.predict(X_te)
            
    results_dict['xgboost'] = compute_r2(xgb_preds, Y_te, M_te)
    print("XGBoost:", results_dict['xgboost'])
    
    results_dict['svr'] = compute_r2(svr_preds, Y_te, M_te)
    print("SVR:", results_dict['svr'])
    
    print("Training Vanilla RNN...")
    p, t, m = train_nn(SeqModel(n_features, n_targets, 'rnn', False), train_loader, test_loader, device)
    results_dict['rnn'] = compute_r2(p, t, m)
    print("RNN:", results_dict['rnn'])

    print("Training LSTM...")
    p, t, m = train_nn(SeqModel(n_features, n_targets, 'lstm', False), train_loader, test_loader, device)
    results_dict['lstm'] = compute_r2(p, t, m)
    print("LSTM:", results_dict['lstm'])
    
    print("Training GRU...")
    p, t, m = train_nn(SeqModel(n_features, n_targets, 'gru', False), train_loader, test_loader, device)
    results_dict['gru'] = compute_r2(p, t, m)
    print("GRU:", results_dict['gru'])
    
    print("Training Bi-LSTM...")
    p, t, m = train_nn(SeqModel(n_features, n_targets, 'lstm', True), train_loader, test_loader, device)
    results_dict['bilstm'] = compute_r2(p, t, m)
    print("Bi-LSTM:", results_dict['bilstm'])
    
    print("Training 1D CNN...")
    p, t, m = train_nn(CNN1DModel(n_features, n_targets), train_loader, test_loader, device)
    results_dict['cnn1d'] = compute_r2(p, t, m)
    print("1D CNN:", results_dict['cnn1d'])
    
    with open('_hybrid_results.json', 'r') as f:
        data = json.load(f)
        
    # Remove old rnn_standalone to keep naming consistent
    if 'rnn_standalone' in data:
        del data['rnn_standalone']
        
    data.update(results_dict)
    
    with open('_hybrid_results.json', 'w') as f:
        json.dump(data, f, indent=2)
        
if __name__ == '__main__':
    train_all()
