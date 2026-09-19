import argparse
import json
import gc
from pathlib import Path
from datetime import datetime
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, f1_score,
    confusion_matrix, brier_score_loss
)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

COLORS = {
    'clinical': '#3C8DBC',
    'image_only': '#00A087',
    'both': '#F39B7F',
    'reference': '#999999',
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def normalize_patient_id(pid, width=8):
    if isinstance(pid, str):
        pid = pid.strip()
        if '.' in pid:
            pid = pid.split('.')[0]
        if pid.isdigit():
            pid = pid.zfill(width)
    return str(pid)

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    brier = brier_score_loss(y_true, y_prob)

    return {
        "accuracy": float(acc),
        "auc": float(auc),
        "f1": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "brier_score": float(brier),
    }

def bootstrap_ci(y_true, y_prob, n_bootstrap=1000, ci=0.95):
    n = len(y_true)
    aucs = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]

        if len(np.unique(y_true_boot)) < 2:
            continue

        try:
            auc = roc_auc_score(y_true_boot, y_prob_boot)
            aucs.append(auc)
        except:
            continue

    if len(aucs) == 0:
        return np.nan, np.nan, np.nan

    alpha = 1 - ci
    lower = np.percentile(aucs, alpha / 2 * 100)
    upper = np.percentile(aucs, (1 - alpha / 2) * 100)
    median = np.median(aucs)

    return float(median), float(lower), float(upper)


def load_clinical_results(cfg):
    stage2_dir = Path(cfg["output_dir"]) / "stage2"
    summary_path = stage2_dir / "stage2_model_summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_mil_results(cfg):
    stage3_dir = Path(cfg["output_dir"]) / "stage3"
    summary_path = stage3_dir / "stage3_model_summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_clinical_models_with_raw(cfg):
    import joblib
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

    stage2_dir = Path(cfg["output_dir"]) / "stage2"
    pat_df = pd.read_csv(cfg["pat_csv"]).copy()
    pat_df = pat_df.dropna(subset=["healing"]).reset_index(drop=True)
    pat_df["patient_id"] = pat_df["patient_id"].astype(str).map(lambda x: x.zfill(8))
    pat_df["drug_enc"] = (pat_df["drug"].astype(str) == "VDZ").astype(np.float32)

    split_info_path = Path(cfg["output_dir"]) / "cache" / "split_info_healing.json"
    split_info = json.load(open(split_info_path, "r", encoding="utf-8"))
    test_ids = set(split_info["test_patient_ids"])
    test_df = pat_df[pat_df["patient_id"].isin(test_ids)].copy()

    ref_model_dir = stage2_dir / "LogisticRegression_single" / "hf_model"
    if ref_model_dir.exists():
        ref_checkpoint = torch.load(ref_model_dir / "model.pt", map_location='cpu')
        feature_cols = ref_checkpoint.get('feature_cols', [])
        print(f"[Clinical] Using {len(feature_cols)} features from model config")
    else:
        print("[Clinical] WARNING: Could not find model config, using default features")
        feature_cols = ['Age', 'BMI', 'MayoScore', 'HCT', 'Hb', 'LYM#', 'M#', 'NE#', 'PLT', 'WBC', 'TP', 'Alb', 'Cr', 'PA', 'D-Dimer', 'PTA', 'PT', 'hsCRP', 'ESR', 'Fib', 'drug_enc']

    available_cols = [c for c in feature_cols if c in pat_df.columns]
    if len(available_cols) != len(feature_cols):
        missing = set(feature_cols) - set(available_cols)
        print(f"[Clinical] Warning: missing columns: {missing}")
    feature_cols = available_cols

    train_ids = set(split_info["train_patient_ids"])
    train_df = pat_df[pat_df["patient_id"].isin(train_ids)].copy()

    train_medians = train_df[feature_cols].median()

    X_test = test_df[feature_cols].fillna(train_medians).values.astype(np.float32)
    y_test = test_df["remission"].values.astype(np.float32)

    results = {}

    model_names = ['LogisticRegression_single', 'RandomForest_single', 'SVM_single', 'XGBoost_single']
    for model_name in model_names:
        model_path = stage2_dir / model_name / f"{model_name}.joblib"
        scaler_path = stage2_dir / model_name / "scaler.joblib"
        if not model_path.exists():
            continue

        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            X_test_scaled = scaler.transform(X_test)
            probs = model.predict_proba(X_test_scaled)[:, 1]

            auc = roc_auc_score(y_test, probs)
            preds = (probs >= 0.5).astype(int)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds)

            results[model_name] = {
                'model_name': model_name.replace('_single', ''),
                'test_metrics': {
                    'auc': auc,
                    'accuracy': acc,
                    'f1': f1,
                },
                'raw': {
                    'remission': {
                        'y': y_test.tolist(),
                        'p': probs.tolist(),
                    }
                }
            }
            print(f"[Clinical] {model_name}: AUC={auc:.4f}, Acc={acc:.4f}, F1={f1:.4f}")
        except Exception as e:
            print(f"[Clinical] {model_name} evaluation failed: {e}")

    mlp_path = stage2_dir / "MLP_drug_condition" / "MLP_drug_condition.pt"
    mlp_hf_path = stage2_dir / "MLP_drug_condition" / "hf_model" / "model.pt"
    scaler_path = stage2_dir / "MLP_drug_condition" / "scaler.joblib"
    if mlp_path.exists() and mlp_hf_path.exists() and scaler_path.exists():
        try:
            checkpoint = torch.load(mlp_path, map_location='cpu')
            hf_checkpoint = torch.load(mlp_hf_path, map_location='cpu')
            mlp_feature_cols = hf_checkpoint.get('feature_cols', [])
            scaler = joblib.load(scaler_path)

            all_feature_cols = mlp_feature_cols + ["drug_enc"]
            train_medians = train_df[all_feature_cols].median()
            X_test_all = test_df[all_feature_cols].fillna(train_medians).values.astype(np.float32)
            X_test_all_scaled = scaler.transform(X_test_all)

            X_test_mlp_scaled = X_test_all_scaled[:, :20]

            class MLPDrugCondition(nn.Module):
                def __init__(self, n_features, hidden=128, dropout=0.3):
                    super().__init__()
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

                def forward(self, x, drug):
                    h = self.shared(x)
                    drug_repr = self.drug_encoder(drug)
                    gate_value = self.gate_net(drug_repr)
                    final_hidden = h * gate_value
                    return torch.sigmoid(self.head_rem(final_hidden)).squeeze(-1)

            model = MLPDrugCondition(len(mlp_feature_cols), hidden=128, dropout=0.3)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            drug_test = test_df["drug_enc"].values.astype(np.float32)

            with torch.no_grad():
                X_tensor = torch.tensor(X_test_mlp_scaled, dtype=torch.float32)
                drug_tensor = torch.tensor(drug_test, dtype=torch.float32).unsqueeze(1)
                probs = model(X_tensor, drug_tensor).numpy()

            auc = roc_auc_score(y_test, probs)
            preds = (probs >= 0.5).astype(int)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds)

            results['MLP_drug_condition'] = {
                'model_name': 'MLP_DrugCondition',
                'test_metrics': {
                    'auc': auc,
                    'accuracy': acc,
                    'f1': f1,
                },
                'raw': {
                    'remission': {
                        'y': y_test.tolist(),
                        'p': probs.tolist(),
                    }
                }
            }
            print(f"[Clinical] MLP_drug_condition: AUC={auc:.4f}, Acc={acc:.4f}, F1={f1:.4f}")
        except Exception as e:
            print(f"[Clinical] MLP_drug_condition evaluation failed: {e}")

    return results

def evaluate_mil_models(cfg, device):
    from stage3_transformer import (
        build_image_model, compute_clinical_stats, ImageBagDataset,
        run_eval_dualhead, normalize_patient_id, GastroNetEncoderWrapper
    )
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms

    stage_cfg = cfg["stage3"]
    stage3_dir = Path(cfg["output_dir"]) / "stage3"

    pat_df = pd.read_csv(cfg["pat_csv"]).copy()
    pat_df = pat_df.dropna(subset=["healing"]).reset_index(drop=True)
    pat_df["patient_id"] = pat_df["patient_id"].astype(str).map(normalize_patient_id)
    pat_df["drug_enc"] = (pat_df["drug"].astype(str) == "VDZ").astype(np.float32)
    img_df = pd.read_csv(cfg["img_csv"]).copy()
    img_df = img_df[img_df["annotated_label"].isin([0, 1])].copy().reset_index(drop=True)

    clinical_cols = list(stage_cfg["clinical_cols"])
    med = pat_df[clinical_cols].median(numeric_only=True)
    pat_df[clinical_cols] = pat_df[clinical_cols].fillna(med)

    split_info_path = Path(cfg["output_dir"]) / "cache" / "split_info_healing.json"
    split_info = json.load(open(split_info_path, "r", encoding="utf-8"))
    train_ids = set(split_info["train_patient_ids"])
    test_ids = set(split_info["test_patient_ids"])
    train_df = pat_df[pat_df["patient_id"].isin(train_ids)].copy()
    test_df = pat_df[pat_df["patient_id"].isin(test_ids)].copy()

    image_size = cfg["stage1"]["image_size"]
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    encoder_ckpt = Path(cfg["gastronet_ckpt"])
    backbone_name = cfg["stage1"]["backbone_name"]
    encoder = GastroNetEncoderWrapper(
        backbone_name, image_size, str(encoder_ckpt),
        unfreeze_last_n_blocks=2, pretrained_dinov2=True
    )
    encoder.to(device)
    encoder.eval()

    clinical_mean, clinical_std = compute_clinical_stats(train_df, clinical_cols)

    results = {}

    for model_name in ["image_only", "both"]:
        print(f"\n[Evaluating] {model_name}...")

        n_drug = 1
        model = build_image_model(model_name, encoder, len(clinical_cols), n_drug, stage_cfg)
        model.to(device)

        ckpt_path = stage3_dir / f"{model_name}_best_model.pt"
        if not ckpt_path.exists():
            print(f"[Warning] Model not found: {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        test_ds = ImageBagDataset(
            test_df, img_df, cfg["image_dir"], eval_tf, clinical_cols,
            mode=model_name, bag_size=stage_cfg.get("bag_size", 100),
            max_images_per_patient=stage_cfg.get("max_images_per_patient", 100),
            training=False, clinical_mean=clinical_mean, clinical_std=clinical_std
        )
        test_loader = DataLoader(test_ds, batch_size=stage_cfg.get("batch_size_bag", 4),
                                shuffle=False, num_workers=0, pin_memory=True)

        test_metrics = run_eval_dualhead(model, test_loader, device, use_amp=True)

        print(f"[{model_name}] Remission: AUC={test_metrics['remission']['auc']:.4f}, "
              f"ACC={test_metrics['remission']['acc']:.4f}, F1={test_metrics['remission']['f1']:.4f}")

        results[model_name] = {
            "best_epoch": ckpt["epoch"],
            "best_rem_auc": test_metrics["remission"]["auc"],
            "best_hea_auc": test_metrics["healing"]["auc"],
            "test_metrics": test_metrics,
        }

    return results


def plot_roc_curves(all_results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = {
        'XGBoost_single': '#4ECDC4',
        'image_only': '#3C8DBC',
        'both': '#E24A33',
    }

    display_models = ['XGBoost_single', 'image_only', 'both']

    for i, model_key in enumerate(display_models):
        ax = axes[i]
        if model_key not in all_results:
            continue
        data = all_results[model_key]
        y_true = data.get('y_true', [])
        y_prob = data.get('y_prob', [])

        if y_true and y_prob and len(np.unique(y_true)) > 1:
            y_true = np.array(y_true)
            y_prob = np.array(y_prob)
            auc = data.get('metrics', {}).get('auc', 0)

            sorted_idx = np.argsort(y_prob)[::-1]
            y_sorted = y_true[sorted_idx]

            n_pos = np.sum(y_true == 1)
            n_neg = np.sum(y_true == 0)

            fpr_points = []
            tpr_points = []

            tp, fp = 0, 0
            for label in y_sorted:
                if label == 1:
                    tp += 1
                else:
                    fp += 1
                fpr_points.append(fp / n_neg)
                tpr_points.append(tp / n_pos)

            fpr_points = [0] + fpr_points
            tpr_points = [0] + tpr_points

            fpr_points = np.array(fpr_points)
            tpr_points = np.array(tpr_points)

            color = colors.get(model_key, '#999999')
            label = model_key.replace('_single', '').replace('_', ' ').title()

            ax.plot(fpr_points, tpr_points, color=color, lw=2, zorder=3)

            ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)

            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.02])
            ax.set_xlabel('False Positive Rate', fontsize=11)
            ax.set_ylabel('True Positive Rate', fontsize=11)
            ax.set_title(f'{label} (AUC={auc:.2f})', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')

    fig.tight_layout()
    fig.savefig(output_dir / "roc_comparison_all_models.svg", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[Saved] {output_dir / 'roc_comparison_all_models.svg'}")

def plot_performance_table(all_results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name, data in all_results.items():
        if 'test_metrics' in data:
            m = data['test_metrics']
            rem_auc = m.get('remission', {}).get('auc', 0)
            rem_acc = m.get('remission', {}).get('acc', 0)
            rem_f1 = m.get('remission', {}).get('f1', 0)
        elif 'metrics' in data:
            m = data['metrics']
            rem_auc = m.get('auc', 0)
            rem_acc = m.get('accuracy', 0)
            rem_f1 = m.get('f1', 0)
        else:
            continue

        rows.append({
            "Model": model_name,
            "AUC": f"{rem_auc:.4f}",
            "Accuracy": f"{rem_acc:.4f}",
            "F1": f"{rem_f1:.4f}",
        })

    if not rows:
        return

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, len(df) * 0.5 + 1))
    ax.axis('off')

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#f0f0f0'] * len(df.columns)
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    best_idx = max(range(len(rows)), key=lambda i: float(rows[i]["AUC"]))
    for j in range(len(df.columns)):
        table[(best_idx + 1, j)].set_facecolor('#90EE90')

    fig.tight_layout()
    fig.savefig(output_dir / "performance_table.svg", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[Saved] {output_dir / 'performance_table.svg'}")


def main():
    parser = argparse.ArgumentParser(description="Stage4: Model Evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()

    cfg = json.load(open(args.config, "r", encoding="utf-8"))
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Stage4] device={device}")

    output_dir = Path(cfg["output_dir"]) / "stage4"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("= Stage 4: Model Evaluation")
    print("=" * 80)

    clinical_results = evaluate_clinical_models_with_raw(cfg)
    stage3_results = evaluate_mil_models(cfg, device)

    all_results = {}

    for model_key, data in clinical_results.items():
        test_metrics = data.get('test_metrics', {})
        raw = data.get('raw', {}).get('remission', {})
        if test_metrics:
            all_results[model_key] = {
                'metrics': test_metrics,
                'y_true': raw.get('y', []),
                'y_prob': raw.get('p', []),
            }

    for model_key in ['image_only', 'both']:
        if model_key not in stage3_results:
            continue
        data = stage3_results[model_key]
        test_metrics = data['test_metrics']
        all_results[model_key] = {
            'metrics': {
                'auc': data['best_rem_auc'],
                'accuracy': test_metrics['remission']['acc'],
                'f1': test_metrics['remission']['f1'],
                'sensitivity': 0,
                'specificity': 0,
                'brier_score': test_metrics['remission'].get('brier', 0),
            },
            'y_true': test_metrics['raw']['remission']['y'],
            'y_prob': test_metrics['raw']['remission']['p'],
            'bootstrap_ci': {},
        }

    print("\n" + "=" * 100)
    print("性能评估表格 (Performance Metrics Table)")
    print("=" * 100)
    print(f"{'Model':<30} | {'AUC':>8} | {'Accuracy':>8} | {'F1':>8}")
    print("-" * 100)

    for model_name, data in all_results.items():
        if 'metrics' in data:
            m = data['metrics']
            auc = m.get('auc', 0)
            acc = m.get('accuracy', 0)
            f1 = m.get('f1', 0)
            print(f"{model_name:<30} | {auc:>8.4f} | {acc:>8.4f} | {f1:>8.4f}")

    print("=" * 100)

    print("\n" + "=" * 80)
    print("Bootstrap CI (95% Confidence Interval)")
    print("=" * 80)
    print(f"{'Model':<30} | {'AUC Median':>10} | {'95% CI Lower':>12} | {'95% CI Upper':>12}")
    print("-" * 70)

    bootstrap_results = {}
    for model_name, data in all_results.items():
        if 'y_true' in data and 'y_prob' in data:
            y_true = np.array(data['y_true'])
            y_prob = np.array(data['y_prob'])
            median, lower, upper = bootstrap_ci(y_true, y_prob, n_bootstrap=args.n_bootstrap)
            bootstrap_results[model_name] = {
                'auc_median': median,
                'ci_lower': lower,
                'ci_upper': upper
            }
            print(f"{model_name:<30} | {median:>10.4f} | {lower:>12.4f} | {upper:>12.4f}")
        elif 'bootstrap_ci' in data:
            ci = data['bootstrap_ci']
            median = ci.get('auc_mean', 0)
            lower = ci.get('ci_lower', 0)
            upper = ci.get('ci_upper', 0)
            bootstrap_results[model_name] = {
                'auc_median': median,
                'ci_lower': lower,
                'ci_upper': upper
            }
            print(f"{model_name:<30} | {median:>10.4f} | {lower:>12.4f} | {upper:>12.4f}")

    print("=" * 80)

    print("\n[Plotting] ROC Curves...")
    plot_roc_curves(all_results, output_dir)

    print("\n[Plotting] Performance Table...")
    plot_performance_table(all_results, output_dir)

    results_path = output_dir / "stage4_results.json"

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
            "results": {k: {
                'metrics': v.get('metrics', {}),
                'bootstrap_ci': bootstrap_results.get(k, {})
            } for k, v in all_results.items()}
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[Results] Saved to {results_path}")
    print("\n" + "=" * 80)
    print("Stage 4 Evaluation Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
