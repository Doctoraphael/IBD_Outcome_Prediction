import argparse
import json
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss, confusion_matrix
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[Warning] XGBoost not installed, skipping")


def normalize_patient_id(pid) -> str:
    s = str(pid).strip()
    if not s:
        return s
    if s.isdigit():
        return s.zfill(8)
    return s

def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def configure_gpu(cfg: dict):
    if not torch.cuda.is_available():
        return torch.device("cpu")
    torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", True))
    return torch.device("cuda")

def find_optimal_threshold(prob, true):
    import scipy.stats as stats

    prob = np.asarray(prob)
    true = np.asarray(true)

    pos_probs = prob[true == 1]
    neg_probs = prob[true == 0]

    if len(pos_probs) == 0 or len(neg_probs) == 0:
        return 0.5, None

    best_thresh, best_separation = 0.5, -1

    for thresh in np.arange(0.1, 0.9, 0.02):
        tpr = np.mean(pos_probs >= thresh)
        fpr = np.mean(neg_probs >= thresh)

        separation = tpr - fpr

        if separation > best_separation:
            best_separation = separation
            best_thresh = thresh

    pred = (prob >= best_thresh).astype(int)
    f1 = f1_score(true, pred, zero_division=0)

    return best_thresh, f1

def compute_single_metrics(true, prob, name="remission", optimal_thresh=None):
    auc = roc_auc_score(true, prob) if len(np.unique(true)) > 1 else np.nan
    brier = brier_score_loss(true, prob)

    if optimal_thresh is not None:
        pred = (prob >= optimal_thresh).astype(int)
        acc = accuracy_score(true, pred)
        f1 = f1_score(true, pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        result = {
            "accuracy": float(acc), "auc": float(auc), "f1": float(f1),
            "sensitivity": float(sens), "specificity": float(spec),
            "brier_score": float(brier),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "optimal_threshold": float(optimal_thresh),
        }
    else:
        pred = (prob >= 0.5).astype(int)
        acc = accuracy_score(true, pred)
        f1 = f1_score(true, pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        result = {
            "accuracy": float(acc), "auc": float(auc), "f1": float(f1),
            "sensitivity": float(sens), "specificity": float(spec),
            "brier_score": float(brier),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }

    return result

def compute_dual_metrics(rem_true, rem_prob, hea_true, hea_prob):
    return compute_single_metrics(rem_true, rem_prob, "remission")


class ClinicalMLPSingle(nn.Module):

    def __init__(self, n_features, hidden=128, dropout=0.3):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.shared = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.head_rem = nn.Linear(hidden // 2, 1)

        self.to(self.device)

    def forward(self, x):
        h = self.shared(x)
        rem_logits = self.head_rem(h).squeeze(-1)
        return rem_logits

class ClinicalMLPDrugCondition(nn.Module):

    def __init__(self, n_features, hidden=128, dropout=0.3):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.shared = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.drug_encoder = nn.Sequential(
            nn.Linear(1, hidden // 4),
            nn.ReLU(),
            nn.Linear(hidden // 4, hidden // 2),
        )

        self.gate_net = nn.Sequential(
            nn.Linear(hidden // 2, hidden // 2),
            nn.Sigmoid(),
        )

        self.head_rem = nn.Linear(hidden // 2, 1)

        self.to(self.device)

    def forward(self, x, drug):
        h = self.shared(x)

        drug_repr = self.drug_encoder(drug)

        gate_value = self.gate_net(drug_repr)
        final_hidden = h * gate_value

        rem_logits = self.head_rem(final_hidden).squeeze(-1)
        return rem_logits

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            else:
                X_tensor = X.to(self.device)
            rem_logits = self(X_tensor)
            rem_prob = torch.sigmoid(rem_logits).cpu().numpy()
        return rem_prob

class SingleHeadTrainer:

    def __init__(self, n_features, hidden=128, dropout=0.3, lr=1e-3, device="cuda"):
        self.device = torch.device(device)
        self.model = ClinicalMLPSingle(n_features, hidden=hidden, dropout=dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()

    def fit(self, X, y_rem, epochs=300, batch_size=32, patience=30, verbose=True):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_rem_tensor = torch.tensor(y_rem, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_rem_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0

            for batch_x, batch_rem in loader:
                self.optimizer.zero_grad()
                rem_logits = self.model(batch_x)
                rem_loss = self.criterion(rem_logits, batch_rem)
                rem_loss.backward()
                self.optimizer.step()
                epoch_loss += rem_loss.item()

            avg_loss = epoch_loss / len(loader)

            if avg_loss < best_loss - 1e-5:
                best_loss = avg_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

    def fit_with_early_stop(self, X, y_rem, X_val, y_val_rem,
                            epochs=300, batch_size=32, patience=30, verbose=True):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_rem_tensor = torch.tensor(y_rem, dtype=torch.float32).to(self.device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_rem_tensor = torch.tensor(y_val_rem, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_rem_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_auc = 0
        best_state = None
        patience_counter = 0
        best_epoch = 0

        for epoch in range(epochs):
            self.model.train()
            for batch_x, batch_rem in loader:
                self.optimizer.zero_grad()
                rem_logits = self.model(batch_x)
                rem_loss = self.criterion(rem_logits, batch_rem)
                rem_loss.backward()
                self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_rem_prob = self.model.predict(X_val_tensor)
                val_auc = roc_auc_score(y_val_rem, val_rem_prob) if len(np.unique(y_val_rem)) > 1 else 0

            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
                best_epoch = epoch + 1
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stop: best_auc={best_auc:.4f} at epoch {best_epoch}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        return best_epoch, best_auc

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        torch.save({"model_state_dict": self.model.state_dict()}, path)

class DrugConditionTrainer:

    def __init__(self, n_features, hidden=128, dropout=0.3, lr=1e-3, device="cuda"):
        self.device = torch.device(device)
        self.model = ClinicalMLPDrugCondition(n_features, hidden=hidden, dropout=dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()

    def fit(self, X, drug, y_rem, epochs=300, batch_size=32, patience=30, verbose=True):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        drug_tensor = torch.tensor(drug, dtype=torch.float32).to(self.device).unsqueeze(1)
        y_rem_tensor = torch.tensor(y_rem, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor, drug_tensor, y_rem_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0

            for batch_x, batch_drug, batch_rem in loader:
                self.optimizer.zero_grad()
                rem_logits = self.model(batch_x, batch_drug)
                rem_loss = self.criterion(rem_logits, batch_rem)
                rem_loss.backward()
                self.optimizer.step()
                epoch_loss += rem_loss.item()

            avg_loss = epoch_loss / len(loader)

            if avg_loss < best_loss - 1e-5:
                best_loss = avg_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

    def fit_with_early_stop(self, X, drug, y_rem, X_val, drug_val, y_val_rem,
                            epochs=300, batch_size=32, patience=30, verbose=True):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        drug_tensor = torch.tensor(drug, dtype=torch.float32).to(self.device).unsqueeze(1)
        y_rem_tensor = torch.tensor(y_rem, dtype=torch.float32).to(self.device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        drug_val_tensor = torch.tensor(drug_val, dtype=torch.float32).to(self.device).unsqueeze(1)
        y_val_rem_tensor = torch.tensor(y_val_rem, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor, drug_tensor, y_rem_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_auc = 0
        best_state = None
        patience_counter = 0
        best_epoch = 0

        for epoch in range(epochs):
            self.model.train()
            for batch_x, batch_drug, batch_rem in loader:
                self.optimizer.zero_grad()
                rem_logits = self.model(batch_x, batch_drug)
                rem_loss = self.criterion(rem_logits, batch_rem)
                rem_loss.backward()
                self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_probs = self.predict(X_val_tensor, drug_val_tensor)
                try:
                    if len(np.unique(y_val_rem)) > 1:
                        val_auc = roc_auc_score(y_val_rem, val_probs)
                    else:
                        val_auc = 0.0
                except Exception:
                    val_auc = 0.0

            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
                best_epoch = epoch + 1
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stop: best_auc={best_auc:.4f} at epoch {best_epoch}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        return best_epoch, best_auc

    def predict(self, X, drug):
        self.model.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            else:
                X_tensor = X.to(self.device)
            if isinstance(drug, np.ndarray):
                drug_tensor = torch.tensor(drug, dtype=torch.float32).to(self.device).unsqueeze(1)
            else:
                drug_tensor = drug.to(self.device).unsqueeze(1)
            rem_logits = self.model(X_tensor, drug_tensor)
            rem_prob = torch.sigmoid(rem_logits).cpu().numpy()
        return rem_prob

    def save(self, path):
        torch.save({"model_state_dict": self.model.state_dict()}, path)


def save_sklearn_huggingface(model, model_name, n_features, feature_cols, out_dir):
    try:
        from safetensors.torch import save_file as safe_save_file
        import joblib

        hf_dir = out_dir / "hf_model"
        hf_dir.mkdir(parents=True, exist_ok=True)

        state_dict = {}
        if model_name == "LogisticRegression":
            state_dict["coef"] = torch.tensor(model.coef_, dtype=torch.float32)
            state_dict["intercept"] = torch.tensor(model.intercept_, dtype=torch.float32)
            safe_save_file(state_dict, hf_dir / "model.safetensors")
        elif model_name == "RandomForest":
            joblib.dump(model, hf_dir / "rf_model.joblib")
            state_dict["n_estimators"] = torch.tensor(model.n_estimators, dtype=torch.int32)
            safe_save_file(state_dict, hf_dir / "model.safetensors")
        elif model_name == "XGBoost":
            joblib.dump(model, hf_dir / "xgb_model.joblib")
            state_dict["n_estimators"] = torch.tensor(model.n_estimators, dtype=torch.int32)
            safe_save_file(state_dict, hf_dir / "model.safetensors")
        elif model_name == "SVM":
            joblib.dump(model, hf_dir / "svm_model.joblib")
            if hasattr(model, 'support_vectors_'):
                state_dict["n_support_vectors"] = torch.tensor(len(model.support_vectors_), dtype=torch.int32)
                safe_save_file(state_dict, hf_dir / "model.safetensors")

        torch.save({"model_name": model_name, "feature_cols": feature_cols}, hf_dir / "model.pt")
        print(f"  [{model_name}] HuggingFace format saved to {hf_dir}")
    except Exception as e:
        print(f"  [{model_name}] HuggingFace format save failed: {e}")


def train_sklearn_cv(model_class, model_params, X, y_rem, skf, model_name):
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_rem)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_rem_train, y_rem_val = y_rem[train_idx], y_rem[val_idx]

        if model_class == "xgb" and HAS_XGBOOST:
            model = XGBClassifier(**model_params, objective='binary:logistic', eval_metric='auc')
            model.fit(X_train, y_rem_train, eval_set=[(X_val, y_rem_val)], verbose=False)
        else:
            model = model_class(**model_params)
            model.fit(X_train, y_rem_train)

        probs = model.predict_proba(X_val)[:, 1]
        metrics = compute_single_metrics(y_rem_val, probs, "remission")
        fold_results.append(metrics)

    return fold_results

def train_final_sklearn(model_class, model_params, X, y_rem, model_name):
    if model_class == "xgb" and HAS_XGBOOST:
        final_params = {k: v for k, v in model_params.items() if k != "early_stopping_rounds"}
        model = XGBClassifier(**final_params, objective='binary:logistic')
        model.fit(X, y_rem, verbose=False)
    else:
        model = model_class(**model_params)
        model.fit(X, y_rem)
    return model


def main():
    parser = argparse.ArgumentParser(description="Stage2: Clinical ML (Single Task: remission)")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = json.load(open(args.config, "r", encoding="utf-8"))
    seed_everything(cfg["seed"])
    device = configure_gpu(cfg)
    use_cuda = device.type == "cuda"

    print("=" * 80)
    print("Stage 2: Clinical ML Models (Single Task: remission, healing as auxiliary)")
    print("=" * 80)

    cache_dir = Path(cfg["output_dir"]) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    pat_df = pd.read_csv(cfg["pat_csv"]).copy()
    pat_df = pat_df.dropna(subset=["remission", "healing"]).reset_index(drop=True)
    pat_df["patient_id"] = pat_df["patient_id"].astype(str).map(normalize_patient_id)
    pat_df["drug_enc"] = (pat_df["drug"].astype(str) == "UST").astype(np.float32)

    clinical_cols = list(cfg["stage23"]["clinical_cols"])
    med = pat_df[clinical_cols].median(numeric_only=True)
    pat_df[clinical_cols] = pat_df[clinical_cols].fillna(med)

    y_rem = pat_df["remission"].values.astype(np.float32)

    print(f"\nLabel distribution:")
    print(f"  remission: 0={sum(y_rem==0)}, 1={sum(y_rem==1)}")

    correlations = []
    for col in clinical_cols:
        corr = abs(np.corrcoef(pat_df[col].values, y_rem)[0,1])
        if not np.isnan(corr):
            correlations.append((col, corr))
    correlations.sort(key=lambda x: x[1], reverse=True)

    n_selected_features = 20
    selected_clinical = [col for col, _ in correlations[:n_selected_features]]
    print(f"\n[Feature Selection] Top {n_selected_features} features by correlation:")
    for col, corr in correlations[:n_selected_features]:
        print(f"  {col}: {corr:.3f}")

    feature_cols = selected_clinical + ["drug_enc"]
    n_features = len(feature_cols)
    n_clinical_features = len(selected_clinical)

    split_info_path = cache_dir / "split_info_healing.json"
    if not split_info_path.exists():
        split_info_path = cache_dir / "split_info_dualhead.json"

    if split_info_path.exists():
        print(f"\n[INFO] 加载统一划分: {split_info_path.name}")
        with open(split_info_path, "r", encoding="utf-8") as f:
            split_info_loaded = json.load(f)
        train_ids = set(split_info_loaded["train_patient_ids"])
        test_ids = set(split_info_loaded["test_patient_ids"])
        train_mask = pat_df["patient_id"].isin(train_ids)
        test_mask = pat_df["patient_id"].isin(test_ids)
        tr_all = pat_df[train_mask].reset_index(drop=True)
        te = pat_df[test_mask].reset_index(drop=True)
    else:
        strat_labels = pat_df["drug_enc"].astype(int).astype(str) + "_" + pat_df["remission"].astype(int).astype(str)
        from sklearn.model_selection import train_test_split
        tr_all, te = train_test_split(pat_df, test_size=cfg["test_size"],
                                       random_state=cfg["train_test_split_seed"], stratify=strat_labels)
        tr_all = tr_all.reset_index(drop=True)
        te = te.reset_index(drop=True)

    print(f"\nTrain: {len(tr_all)}, Test: {len(te)}")

    X_train = tr_all[feature_cols].values.astype(np.float32)
    y_rem_train = tr_all["remission"].values.astype(np.float32)
    y_hea_train = tr_all["healing"].values.astype(np.float32)
    drug_train = tr_all["drug_enc"].values.astype(np.float32)
    X_test = te[feature_cols].values.astype(np.float32)
    y_rem_test = te["remission"].values.astype(np.float32)
    y_hea_test = te["healing"].values.astype(np.float32)
    drug_test = te["drug_enc"].values.astype(np.float32)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    val_size = cfg["stage23"].get("val_size", 0.2)
    strat_labels = tr_all["drug_enc"].astype(int).astype(str) + "_" + tr_all["remission"].astype(int).astype(str)
    from sklearn.model_selection import train_test_split
    tr_sub, val_df = train_test_split(
        tr_all, test_size=val_size, random_state=cfg["seed"], stratify=strat_labels
    )
    tr_sub = tr_sub.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    X_tr_sub = tr_sub[feature_cols].values.astype(np.float32)
    y_tr_sub = tr_sub["remission"].values.astype(np.float32)
    drug_tr_sub = tr_sub["drug_enc"].values.astype(np.float32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df["remission"].values.astype(np.float32)
    drug_val = val_df["drug_enc"].values.astype(np.float32)

    X_tr_sub_scaled = scaler.transform(X_tr_sub)
    X_val_scaled = scaler.transform(X_val)

    print(f"\n[Hold-out Split] Train={len(tr_sub)}, Val={len(val_df)}")

    models_config = {
        "LogisticRegression": {
            "class": LogisticRegression,
            "params": {"C": 0.1, "solver": "lbfgs", "max_iter": 1000, "class_weight": "balanced", "random_state": cfg["seed"]},
        },
        "RandomForest": {
            "class": RandomForestClassifier,
            "params": {"n_estimators": 200, "max_depth": 3, "min_samples_split": 10, "min_samples_leaf": 4, "class_weight": "balanced", "random_state": cfg["seed"], "n_jobs": -1},
        },
        "SVM": {
            "class": SVC,
            "params": {"C": 0.1, "kernel": "rbf", "gamma": "scale", "probability": True, "class_weight": "balanced", "random_state": cfg["seed"]},
        },
    }

    if HAS_XGBOOST:
        models_config["XGBoost"] = {
            "class": "xgb",
            "params": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.7, "colsample_bytree": 0.5, "min_child_weight": 5, "reg_alpha": 0.5, "reg_lambda": 2.0, "random_state": cfg["seed"], "early_stopping_rounds": 30, "n_jobs": -1},
        }

    summary = {}

    print(f"\n{'='*60}")
    print("Clinical Models (Remission Prediction) - Hold-out Strategy")
    print(f"{'='*60}")

    for model_name, model_config in models_config.items():
        print(f"\n[{model_name}]")

        model_class = model_config["class"]

        if model_class == "xgb" and HAS_XGBOOST:
            model = XGBClassifier(**model_config["params"], objective='binary:logistic', eval_metric='auc')
            model.fit(X_tr_sub_scaled, y_tr_sub, eval_set=[(X_val_scaled, y_val)], verbose=False)
        else:
            model = model_class(**model_config["params"])
            model.fit(X_tr_sub_scaled, y_tr_sub)

        val_probs = model.predict_proba(X_val_scaled)[:, 1]
        val_metrics = compute_single_metrics(y_val, val_probs, "remission")
        opt_thresh, opt_f1 = find_optimal_threshold(val_probs, y_val)
        print(f"  Val: Rem AUC={val_metrics['auc']:.3f}, Acc={val_metrics['accuracy']:.3f}, F1={val_metrics['f1']:.3f}, Brier={val_metrics['brier_score']:.3f}")
        print(f"       Optimal thresh={opt_thresh:.2f} → F1={opt_f1:.3f}")

        final_model = train_final_sklearn(model_config["class"], model_config["params"],
                                          X_train_scaled, y_rem_train, model_name)

        probs_test = final_model.predict_proba(X_test_scaled)[:, 1]
        test_metrics = compute_single_metrics(y_rem_test, probs_test, "remission", optimal_thresh=opt_thresh)

        out_dir = Path(cfg["output_dir"]) / "stage2" / f"{model_name}_single"
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, out_dir / f"{model_name}_single.joblib")
        joblib.dump(scaler, out_dir / "scaler.joblib")

        save_sklearn_huggingface(final_model, model_name, n_features, feature_cols, out_dir)

        summary[f"{model_name}_single"] = {
            "model_name": model_name, "task": "single",
            "val_metrics": val_metrics, "test_metrics": test_metrics,
        }

        print(f"  Test: Rem AUC={test_metrics['auc']:.3f}, Acc={test_metrics['accuracy']:.3f}, F1={test_metrics['f1']:.3f}, Brier={test_metrics['brier_score']:.3f} (thresh={opt_thresh:.2f})")
        print(f"         Sens={test_metrics['sensitivity']:.3f}, Spec={test_metrics['specificity']:.3f}, TP={test_metrics['tp']}, FP={test_metrics['fp']}, TN={test_metrics['tn']}, FN={test_metrics['fn']}")

    print(f"\n[MLP_DrugCondition]")
    device_str = "cuda" if use_cuda else "cpu"

    trainer_val = DrugConditionTrainer(n_clinical_features, hidden=128, dropout=0.5, lr=1e-3, device=device_str)
    best_epoch, best_auc = trainer_val.fit_with_early_stop(
        X_tr_sub_scaled[:, :n_clinical_features], drug_tr_sub, y_tr_sub,
        X_val_scaled[:, :n_clinical_features], drug_val, y_val,
        epochs=300, batch_size=32, patience=30, verbose=True
    )
    print(f"  Best epoch={best_epoch}, Val AUC={best_auc:.3f}")

    val_probs_mlp = trainer_val.predict(X_val_scaled[:, :n_clinical_features], drug_val)
    opt_thresh_mlp, opt_f1_mlp = find_optimal_threshold(val_probs_mlp, y_val)
    print(f"  Optimal thresh={opt_thresh_mlp:.2f} → Val F1={opt_f1_mlp:.3f}")

    trainer = DrugConditionTrainer(n_clinical_features, hidden=128, dropout=0.5, lr=1e-3, device=device_str)
    trainer.fit(X_train_scaled[:, :n_clinical_features], drug_train, y_rem_train, epochs=best_epoch, batch_size=32, patience=100, verbose=False)
    rem_prob_test = trainer.predict(X_test_scaled[:, :n_clinical_features], drug_test)
    test_metrics_mlp = compute_single_metrics(y_rem_test, rem_prob_test, "remission", optimal_thresh=opt_thresh_mlp)

    out_dir = Path(cfg["output_dir"]) / "stage2" / "MLP_drug_condition"
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save(out_dir / "MLP_drug_condition.pt")
    joblib.dump(scaler, out_dir / "scaler.joblib")

    try:
        from safetensors.torch import save_file as safe_save_file
        hf_dir = out_dir / "hf_model"
        hf_dir.mkdir(parents=True, exist_ok=True)
        state_dict = {k: v for k, v in trainer.model.state_dict().items()}
        safe_save_file(state_dict, hf_dir / "model.safetensors")
        torch.save({"model_name": "MLP_drug_condition", "feature_cols": selected_clinical}, hf_dir / "model.pt")
        print(f"  [MLP_DrugCondition] HuggingFace format saved to {hf_dir}")
    except ImportError:
        print(f"  [MLP_DrugCondition] safetensors not installed, skipping HuggingFace format")
    except Exception as e:
        print(f"  [MLP_DrugCondition] HuggingFace format save failed: {e}")

    summary["MLP_drug_condition"] = {
        "model_name": "MLP_DrugCondition", "task": "single_drug_condition",
        "best_epoch": best_epoch, "val_auc": best_auc,
        "test_metrics": test_metrics_mlp,
    }

    print(f"  Test: Rem AUC={test_metrics_mlp['auc']:.3f}, Acc={test_metrics_mlp['accuracy']:.3f}, F1={test_metrics_mlp['f1']:.3f}, Brier={test_metrics_mlp['brier_score']:.3f} (thresh={opt_thresh_mlp:.2f})")
    print(f"         Sens={test_metrics_mlp['sensitivity']:.3f}, Spec={test_metrics_mlp['specificity']:.3f}, TP={test_metrics_mlp['tp']}, FP={test_metrics_mlp['fp']}, TN={test_metrics_mlp['tn']}, FN={test_metrics_mlp['fn']}")

    summary_path = Path(cfg["output_dir"]) / "stage2" / "stage2_model_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Stage2 完成! 结果已保存到: {summary_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
