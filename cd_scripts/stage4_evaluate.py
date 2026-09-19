import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, brier_score_loss, confusion_matrix
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

COLORS = {
    'LogisticRegression': '#3C8DBC',
    'RandomForest': '#00A087',
    'XGBoost': '#F39B7F',
    'SVM': '#E64B35',
    'MLP_drug_condition': '#6014BC',
    'image_only': '#00A087',
    'both': '#E64B35',
}

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def normalize_patient_id(pid) -> str:
    s = str(pid).strip()
    if not s:
        return s
    if s.isdigit():
        return s.zfill(8)
    return s

def bootstrap_ci(y_true, y_prob, metric_func, n_bootstrap=2000, ci=0.95):
    np.random.seed(42)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        if len(np.unique(y_true[indices])) < 2:
            scores.append(np.nan)
        else:
            scores.append(metric_func(y_true[indices], y_prob[indices]))
    scores = np.array(scores)
    scores = scores[~np.isnan(scores)]
    if len(scores) == 0:
        return 0.0, 0.0, 0.0
    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    upper = np.percentile(scores, (1 + ci) / 2 * 100)
    return float(np.median(scores)), float(lower), float(upper)

def compute_sensitivity_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sens, spec


def evaluate_clinical_models_with_raw(cfg):
    stage2_dir = Path(cfg["output_dir"]) / "stage2"

    pat_df = pd.read_csv(cfg["pat_csv"]).copy()
    pat_df = pat_df.dropna(subset=["remission", "healing"]).reset_index(drop=True)
    pat_df["patient_id"] = pat_df["patient_id"].astype(str).map(normalize_patient_id)
    pat_df["drug_enc"] = (pat_df["drug"].astype(str) == "UST").astype(np.float32)

    split_info_path = Path(cfg["output_dir"]) / "cache" / "split_info_healing.json"
    split_info = json.load(open(split_info_path, "r", encoding="utf-8"))
    train_ids = set(split_info["train_patient_ids"])
    test_ids = set(split_info["test_patient_ids"])

    top20_clinical_cols = [
        "Age", "HBI", "M#", "PA", "HCT", "Gender", "NE#", "Fib",
        "ESR", "WBC", "Hb", "TP", "Alb", "BMI", "PLT", "PTA",
        "PT", "D-Dimer", "hsCRP", "Cr"
    ]
    feature_cols = top20_clinical_cols + ["drug_enc"]

    med = pat_df[top20_clinical_cols].median(numeric_only=True)
    pat_df[top20_clinical_cols] = pat_df[top20_clinical_cols].fillna(med)

    test_df = pat_df[pat_df["patient_id"].isin(test_ids)].copy()
    y_test = test_df["remission"].values.astype(int)

    print(f"[Clinical] Test patients: {len(test_df)}, pos={sum(y_test)}, neg={len(y_test)-sum(y_test)}")

    results = {}

    model_names = ['LogisticRegression_single', 'RandomForest_single', 'SVM_single', 'XGBoost_single']
    for model_name in model_names:
        model_path = stage2_dir / model_name / f"{model_name}.joblib"
        scaler_path = stage2_dir / model_name / "scaler.joblib"
        if not model_path.exists():
            print(f"[Clinical] {model_name}: model not found, skipping")
            continue

        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            X_test = test_df[feature_cols].values.astype(np.float32)
            X_test_scaled = scaler.transform(X_test)
            probs = model.predict_proba(X_test_scaled)[:, 1]

            auc = roc_auc_score(y_test, probs)
            preds = (probs >= 0.5).astype(int)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, zero_division=0)
            sens, spec = compute_sensitivity_specificity(y_test, preds)
            brier = brier_score_loss(y_test, probs)

            results[model_name] = {
                'model_name': model_name.replace('_single', ''),
                'test_metrics': {
                    'auc': auc,
                    'accuracy': acc,
                    'f1': f1,
                    'sensitivity': sens,
                    'specificity': spec,
                    'brier_score': brier,
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
            train_df = pat_df[pat_df["patient_id"].isin(train_ids)].copy()
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
            f1 = f1_score(y_test, preds, zero_division=0)
            sens, spec = compute_sensitivity_specificity(y_test, preds)
            brier = brier_score_loss(y_test, probs)

            results['MLP_drug_condition'] = {
                'model_name': 'MLP_DrugCondition',
                'test_metrics': {
                    'auc': auc,
                    'accuracy': acc,
                    'f1': f1,
                    'sensitivity': sens,
                    'specificity': spec,
                    'brier_score': brier,
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
        run_eval_dualhead, normalize_patient_id, GastroNetEncoderWrapper,
        find_optimal_threshold
    )

    stage_cfg = cfg["stage23"]
    stage3_dir = Path(cfg["output_dir"]) / "stage3"

    pat_df = pd.read_csv(cfg["pat_csv"]).copy()
    pat_df = pat_df.dropna(subset=["remission", "healing"]).reset_index(drop=True)
    pat_df["patient_id"] = pat_df["patient_id"].astype(str).map(normalize_patient_id)
    pat_df["drug_enc"] = (pat_df["drug"].astype(str) == "UST").astype(np.float32)
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

    print(f"[MIL] Test patients: {len(test_df)}, pos={sum(test_df['remission'])}, neg={len(test_df)-sum(test_df['remission'])}")

    image_size = cfg["stage1"]["image_size"]
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    encoder_ckpt = Path(cfg["gastronet_ckpt"])
    backbone_name = "vit_base_patch14_reg4_dinov2"
    unfreeze_blocks = cfg.get("stage23", {}).get("unfreeze_encoder_blocks", 2)
    encoder = GastroNetEncoderWrapper(
        backbone_name, image_size, str(encoder_ckpt),
        unfreeze_last_n_blocks=unfreeze_blocks, pretrained_dinov2=True
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

        train_ds = ImageBagDataset(
            train_df, img_df, cfg["image_dir"], eval_tf, clinical_cols,
            mode=model_name, bag_size=stage_cfg.get("bag_size", 100),
            max_images_per_patient=stage_cfg.get("max_images_per_patient", 100),
            training=False, clinical_mean=clinical_mean, clinical_std=clinical_std
        )
        train_loader = DataLoader(train_ds, batch_size=stage_cfg.get("batch_size_bag", 4),
                                shuffle=False, num_workers=0, pin_memory=True)
        train_metrics = run_eval_dualhead(model, train_loader, device, use_amp=True)

        train_y = np.array(train_metrics["raw"]["remission"]["y"])
        train_p = np.array(train_metrics["raw"]["remission"]["p"])
        opt_thresh, _ = find_optimal_threshold(train_y, train_p, n_steps=20)
        print(f"[{model_name}] Optimal threshold: {opt_thresh:.4f} (from train set)")

        y_true = np.array(test_metrics["raw"]["remission"]["y"])
        y_prob = np.array(test_metrics["raw"]["remission"]["p"])
        y_pred = (y_prob >= opt_thresh).astype(int)
        sens, spec = compute_sensitivity_specificity(y_true, y_pred)
        brier = brier_score_loss(y_true, y_prob)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        test_metrics["remission"]["sensitivity"] = sens
        test_metrics["remission"]["specificity"] = spec
        test_metrics["remission"]["brier"] = brier
        test_metrics["remission"]["acc"] = acc
        test_metrics["remission"]["f1"] = f1
        test_metrics["remission"]["optimal_threshold"] = opt_thresh

        print(f"[{model_name}] Remission: AUC={test_metrics['remission']['auc']:.4f}, "
              f"ACC={acc:.4f}, F1={f1:.4f}, Sens={sens:.4f}, Spec={spec:.4f} (thresh={opt_thresh:.4f})")

        results[model_name] = {
            "best_epoch": ckpt["epoch"],
            "best_rem_auc": test_metrics["remission"]["auc"],
            "best_hea_auc": test_metrics["healing"]["auc"],
            "test_metrics": test_metrics,
        }

    return results


def generate_attention_maps(cfg, device, output_dir, n_samples=8):
    from stage3_transformer import (
        build_image_model, compute_clinical_stats, ImageBagDataset,
        normalize_patient_id, GastroNetEncoderWrapper
    )
    from PIL import Image
    import torchvision

    stage_cfg = cfg["stage23"]
    stage3_dir = Path(cfg["output_dir"]) / "stage3"

    pat_df = pd.read_csv(cfg["pat_csv"]).copy()
    pat_df = pat_df.dropna(subset=["remission", "healing"]).reset_index(drop=True)
    pat_df["patient_id"] = pat_df["patient_id"].astype(str).map(normalize_patient_id)
    pat_df["drug_enc"] = (pat_df["drug"].astype(str) == "UST").astype(np.float32)
    img_df = pd.read_csv(cfg["img_csv"]).copy()
    img_df = img_df[img_df["annotated_label"].isin([0, 1])].copy().reset_index(drop=True)

    clinical_cols = list(stage_cfg["clinical_cols"])
    med = pat_df[clinical_cols].median(numeric_only=True)
    pat_df[clinical_cols] = pat_df[clinical_cols].fillna(med)

    split_info_path = Path(cfg["output_dir"]) / "cache" / "split_info_healing.json"
    split_info = json.load(open(split_info_path, "r", encoding="utf-8"))
    test_ids = set(split_info["test_patient_ids"])
    test_df = pat_df[pat_df["patient_id"].isin(test_ids)].copy()

    image_size = cfg["stage1"]["image_size"]
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    encoder_ckpt = Path(cfg["gastronet_ckpt"])
    backbone_name = "vit_base_patch14_reg4_dinov2"
    unfreeze_blocks = cfg.get("stage23", {}).get("unfreeze_encoder_blocks", 2)
    encoder = GastroNetEncoderWrapper(
        backbone_name, image_size, str(encoder_ckpt),
        unfreeze_last_n_blocks=unfreeze_blocks, pretrained_dinov2=True
    )
    encoder.to(device)
    encoder.eval()

    clinical_mean, clinical_std = compute_clinical_stats(test_df, clinical_cols)

    attn_dir = output_dir / "attention_maps"
    attn_dir.mkdir(parents=True, exist_ok=True)

    for model_name in ["image_only", "both"]:
        print(f"\n[Attention] Generating for {model_name}...")

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

        all_labels = [(i, test_ds[i]["remission"].item()) for i in range(len(test_ds))]
        pos_indices = [i for i, label in all_labels if label == 1][:n_samples // 2]
        neg_indices = [i for i, label in all_labels if label == 0][:n_samples // 2]
        sample_indices = pos_indices + neg_indices

        model_attn_dir = attn_dir / model_name
        model_attn_dir.mkdir(parents=True, exist_ok=True)

        for idx in sample_indices:
            try:
                item = test_ds[idx]
                images = item["bag_images"]
                clinical = item["clinical"]
                label = item["remission"].item()
                patient_id = item["patient_id"]

                n_valid = item["n_valid"].item()

                if n_valid > 0:
                    img_tensor = images[0:1].to(device)
                else:
                    continue
                clinical_tensor = clinical.unsqueeze(0).to(device)

                with torch.no_grad():
                    if model_name == "both":
                        img_feat = model.encoder(img_tensor)
                        img_tokens = model.img_proj(img_feat)
                        cls_token = model.cls_token.unsqueeze(0).to(device)
                        clinical_feat = model.clinical_encoder(clinical_tensor)
                        clinical_tokens = clinical_feat.unsqueeze(1)
                        tokens = torch.cat([cls_token, img_tokens, clinical_tokens], dim=1)
                        attn_weights = model.transformer_encoder(tokens)
                        attn = attn_weights[0, 0, 1:].cpu().numpy()
                    else:
                        img_feat = model.encoder(img_tensor)
                        img_tokens = model.img_proj(img_feat)
                        cls_token = model.cls_token.unsqueeze(0).to(device)
                        tokens = torch.cat([cls_token, img_tokens], dim=1)
                        attn_weights = model.transformer_encoder(tokens)
                        attn = attn_weights[0, 0, 1:].cpu().numpy()

                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                orig_tensor = images[0] * std + mean
                orig_np = orig_tensor.permute(1, 2, 0).numpy()
                orig_np = (np.clip(orig_np, 0, 1) * 255).astype(np.uint8)

                grid_size = int(np.sqrt(len(attn)))
                attn_map = attn[:grid_size*grid_size].reshape(grid_size, grid_size)
                attn_resized = Image.fromarray(attn_map).resize((orig_np.shape[1], orig_np.shape[0]), Image.BILINEAR)
                attn_np = np.array(attn_resized)
                attn_np = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)

                cmap = plt.cm.jet
                heatmap = cmap(attn_np)[:, :, :3]
                heatmap = (heatmap * 255).astype(np.uint8)

                overlay = (orig_np * 0.6 + heatmap * 0.4).astype(np.uint8)

                save_path = model_attn_dir / f"pat{patient_id}_label{int(label)}_attn.png"
                Image.fromarray(overlay).save(str(save_path))
                print(f"  Saved: {save_path.name}")
            except Exception as e:
                print(f"[Attention] Failed for idx {idx}: {e}")

    print(f"[Attention] Saved to {attn_dir}")


def plot_roc_curves(all_results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    clinical_models = ['LogisticRegression_single', 'RandomForest_single', 'SVM_single', 'XGBoost_single', 'MLP_drug_condition']
    best_clinical = None
    best_clinical_auc = 0
    for model_key in clinical_models:
        if model_key in all_results:
            auc = all_results[model_key].get('metrics', {}).get('auc', 0)
            if auc > best_clinical_auc:
                best_clinical_auc = auc
                best_clinical = model_key

    colors = {
        'Best_Clinical': '#4ECDC4',
        'image_only': '#3C8DBC',
        'both': '#E24A33',
    }

    display_models = [best_clinical, 'image_only', 'both']

    for i, model_key in enumerate(display_models):
        ax = axes[i]
        if model_key is None or model_key not in all_results:
            ax.axis('off')
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
            if model_key == best_clinical:
                label_name = 'Best Clinical'
            else:
                label_name = model_key.replace('_', ' ').title()

            ax.plot(fpr_points, tpr_points, color=color, lw=2, zorder=3)
            ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)

            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.02])
            ax.set_xlabel('False Positive Rate', fontsize=11)
            ax.set_ylabel('True Positive Rate', fontsize=11)
            ax.set_title(f'{label_name} (AUC={auc:.2f})', fontsize=12, fontweight='bold')
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
        if 'metrics' in data:
            m = data['metrics']
            rows.append({
                "Model": model_name.replace('_single', '').replace('_', ' ').title(),
                "AUC": f"{m.get('auc', 0):.4f}",
                "Accuracy": f"{m.get('accuracy', 0):.4f}",
                "F1": f"{m.get('f1', 0):.4f}",
                "Sens": f"{m.get('sensitivity', 0):.4f}",
                "Spec": f"{m.get('specificity', 0):.4f}",
            })

    if not rows:
        return

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, len(df) * 0.6 + 1))
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
    parser = argparse.ArgumentParser(description="Stage4: Model Evaluation (CD)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--skip-attention", action="store_true", help="Skip attention map generation")
    args = parser.parse_args()

    cfg = json.load(open(args.config, "r", encoding="utf-8"))
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Stage4] device={device}")

    output_dir = Path(cfg["output_dir"]) / "stage4"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("= Stage 4: Complete Model Evaluation (CD)")
    print("=" * 80)

    print("\n" + "-" * 60)
    print("Clinical Models Evaluation")
    print("-" * 60)
    clinical_results = evaluate_clinical_models_with_raw(cfg)

    print("\n" + "-" * 60)
    print("MIL Models Evaluation")
    print("-" * 60)
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
                'sensitivity': test_metrics['remission'].get('sensitivity', 0),
                'specificity': test_metrics['remission'].get('specificity', 0),
                'brier_score': test_metrics['remission'].get('brier', 0),
            },
            'y_true': test_metrics['raw']['remission']['y'],
            'y_prob': test_metrics['raw']['remission']['p'],
        }

    print("\n" + "=" * 100)
    print("Performance Metrics Table (CD)")
    print("=" * 100)
    print(f"{'Model':<30} | {'AUC':>8} | {'Acc':>8} | {'F1':>8} | {'Sens':>8} | {'Spec':>8}")
    print("-" * 100)

    for model_name, data in all_results.items():
        if 'metrics' in data:
            m = data['metrics']
            auc = m.get('auc', 0)
            acc = m.get('accuracy', 0)
            f1 = m.get('f1', 0)
            sens = m.get('sensitivity', 0)
            spec = m.get('specificity', 0)
            print(f"{model_name:<30} | {auc:>8.4f} | {acc:>8.4f} | {f1:>8.4f} | {sens:>8.4f} | {spec:>8.4f}")

    print("=" * 100)

    print("\n" + "=" * 70)
    print("Bootstrap CI (95% Confidence Interval)")
    print("=" * 70)
    print(f"{'Model':<30} | {'AUC Median':>10} | {'95% CI Lower':>12} | {'95% CI Upper':>12}")
    print("-" * 70)

    bootstrap_results = {}
    for model_name, data in all_results.items():
        if 'y_true' in data and 'y_prob' in data:
            y_true = np.array(data['y_true'])
            y_prob = np.array(data['y_prob'])
            if len(np.unique(y_true)) < 2:
                print(f"{model_name:<30} | Skip (only one class)")
                bootstrap_results[model_name] = {'auc_median': 0, 'ci_lower': 0, 'ci_upper': 0}
                continue
            median, lower, upper = bootstrap_ci(y_true, y_prob, lambda y, p: roc_auc_score(y, p), args.n_bootstrap)
            bootstrap_results[model_name] = {
                'auc_median': median,
                'ci_lower': lower,
                'ci_upper': upper
            }
            print(f"{model_name:<30} | {median:>10.4f} | {lower:>12.4f} | {upper:>12.4f}")

    print("=" * 70)

    print("\n[Plotting] ROC Curves...")
    plot_roc_curves(all_results, output_dir)

    print("[Plotting] Performance Table...")
    plot_performance_table(all_results, output_dir)

    if not args.skip_attention:
        print("\n[Attention] Generating attention maps...")
        generate_attention_maps(cfg, device, output_dir, n_samples=12)

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
