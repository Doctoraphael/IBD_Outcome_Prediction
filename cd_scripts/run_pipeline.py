import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

def normalize_patient_id(pid) -> str:
    s = str(pid).strip()
    if not s:
        return s
    if s.isdigit():
        return s.zfill(8)
    return s

def build_base_config(root: str) -> dict:
    root = str(Path(root))
    return {
        "root": root,
        "image_dir": str(Path(root) / "imageCD"),
        "img_csv": str(Path(root) / "cd_image_analysis.csv"),
        "pat_csv": str(Path(root) / "cd_patient_summary.csv"),
        "gastronet_ckpt": str(Path(root) / "hf_cache" / "dinov2.pth"),
        "gastronet_model_name": "vit_base_patch14_reg4_dinov2",
        "output_dir": str(Path(root) / "outputs_cd_gastronet5m"),

        "seed": 42,
        "device": "cuda",
        "num_workers": 4,
        "prefetch_factor": 2,
        "pin_memory": True,
        "persistent_workers": True,
        "allow_tf32": True,
        "use_amp": True,
        "channels_last": True,
        "cudnn_benchmark": True,
        "train_test_split_seed": 24,
        "test_size": 0.20,

        "stage1": {
            "backbone_name": "vit_base_patch14_reg4_dinov2",
            "image_size": 336,
            "patch_size": 14,
            "grid_size": 24,
            "num_register_tokens": 4,
            "feature_dim": 768,

            "batch_size": 48,
            "epochs": 140,
            "min_epochs": 40,
            "max_epochs": 220,
            "eval_every": 10,
            "early_stop_patience": 25,

            "lr": 1e-5,
            "head_lr": 8e-4,
            "weight_decay": 1e-4,
            "label_smoothing": 0.02,
            "max_grad_norm": 1.0,

            "freeze_backbone_epochs": 3,

            "auc_gate": 0.70,
            "abnormal_logit_bias": 0.15,

            "feature_batch_size": 96,

            "feature_h5_name": "stage1_patient_bags_gastronet5m.h5",
            "encoder_ckpt_name": "stage1_gastronet5m_encoder.pt",

            "num_workers": 4,
            "prefetch_factor": 2,
            "feature_num_workers": 2,
            "feature_prefetch_factor": 2,
            "gpu_normalize": True
        },

        "stage23": {
            "n_cv_folds": 5,
            "cv_max_epochs": 80,
            "cv_early_stop_patience": 15,

            "epochs": 80,
            "min_epochs": 25,

            "batch_size_bag": 4,
            "bag_size": 100,
            "max_images_per_patient": 100,
            "gradient_accumulation_steps": 1,
            "pos_per_batch": 2,
            "use_checkpoint": True,

            "warmup_epochs": 5,
            "lr": 1e-5,
            "encoder_lr": 5e-6,
            "clinical_encoder_lr": 1e-6,
            "weight_decay": 0.05,
            "gradient_clip": 0.5,

            "dropout": 0.25,
            "clinical_hidden": 64,
            "fusion_hidden": 256,
            "label_smoothing": 0.015,
            "instance_dropout": 0.1,
            "modality_dropout": 0.1,
            "aux_image_loss_weight": 0.02,
            "rank_weight": 0.3,
            "clustering_weight": 0.005,
            "unfreeze_encoder_blocks": 2,
            "d_model": 384,
            "n_heads": 2,
            "n_layers": 1,
            "dim_ffn": 384,

            "mixup_alpha": 0.2,
            "mixup_prob": 0.5,

            "val_size": 0.20,

            "clinical_cols": [
                "Gender", "Age", "BMI", "HBI",
                "HCT", "Hb", "LYM#", "M#", "NE#", "PLT", "WBC",
                "TP", "Alb", "Cr", "PA",
                "D-Dimer", "PTA", "PT", "APTT",
                "hsCRP", "ESR", "Fib"
            ],
            "targets": ["remission", "healing"],

            "model_names": [
                "image_only",
                "both"
            ],

            "num_workers_cli": 0,
            "prefetch_factor_cli": None,
            "num_workers_ima": 0,
            "prefetch_factor_ima": 2,
        },

        "stage4": {
            "n_bootstrap": 2000,
            "cam_examples_per_class": 6,
            "fig_dpi": 180,
            "visualization": {
                "enable": True,
                "n_gradcam_samples": 8,
                "n_attention_samples": 12,
            }
        }
    }

def run_script(script_path: Path, config_path: Path, extra_args=None) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    cmd = [sys.executable, str(script_path), "--config", str(config_path)]
    if extra_args:
        cmd.extend(extra_args)

    print("\n" + "=" * 80)
    print(f"[RUN] {' '.join(cmd)}")
    print("=" * 80 + "\n")

    subprocess.check_call(cmd)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 4-stage CD pipeline with GastroNet-5M backbone for biological agent outcome prediction"
    )
    parser.add_argument(
        "--root",
        default=r"C:\crohn",
        help="项目根目录，包含imageCD文件夹和CSV文件"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录路径，默认为{root}/outputs_cd_gastronet5m"
    )
    parser.add_argument(
        "--config-out",
        default=None,
        help="配置文件输出路径"
    )
    parser.add_argument(
        "--skip-stage1",
        action="store_true",
        help="跳过Stage1 (如果encoder已存在)"
    )
    parser.add_argument(
        "--skip-stage2",
        action="store_true",
        help="跳过Stage2 (临床机器学习模型)"
    )
    parser.add_argument(
        "--skip-stage3",
        action="store_true",
        help="跳过Stage3"
    )
    parser.add_argument(
        "--skip-stage4",
        action="store_true",
        help="跳过Stage4"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    print("=" * 80)
    print("CD生物制剂疗效预测模型训练 pipeline")
    print("GastroNet-5M + TransformerDualHead")
    print("=" * 80)
    print(f"[INFO] 项目根目录: {args.root}")

    cfg = build_base_config(args.root)

    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config_out) if args.config_out else out_dir / "cd_pipeline_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 配置文件: {config_path}")
    print(f"[INFO] 输出目录: {out_dir}")

    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    split_info_path = cache_dir / "split_info_healing.json"

    if not split_info_path.exists():
        print("\n" + "=" * 60)
        print("创建统一数据划分 (drug × healing 分层)")
        print("=" * 60)

        pat_df = pd.read_csv(cfg["pat_csv"])
        pat_df = pat_df.dropna(subset=["healing"]).reset_index(drop=True)
        pat_df["patient_id"] = pat_df["patient_id"].astype(str).map(normalize_patient_id)
        pat_df["drug_enc"] = (pat_df["drug"].astype(str) == "UST").astype(float)

        strat_labels = pat_df["drug_enc"].astype(int).astype(str) + "_" + pat_df["remission"].astype(int).astype(str)
        from sklearn.model_selection import train_test_split
        train_idx, test_idx = train_test_split(
            pat_df.index.tolist(),
            test_size=cfg.get("test_size", 0.20),
            random_state=cfg["seed"],
            stratify=strat_labels
        )

        train_ids = pat_df.iloc[train_idx]["patient_id"].tolist()
        test_ids = pat_df.iloc[test_idx]["patient_id"].tolist()

        strat_df = pat_df.iloc[train_idx].copy()
        strat_stats = {}
        for stratum in strat_df["drug_enc"].astype(int).astype(str) + "_" + strat_df["remission"].astype(int).astype(str):
            strat_stats[stratum] = strat_stats.get(stratum, 0) + 1

        split_info = {
            "task": "dualhead_remission_stratified",
            "stratification": "drug × remission (4-strata)",
            "strat_train_counts": strat_stats,
            "train_patient_ids": train_ids,
            "test_patient_ids": test_ids,
        }
        with open(split_info_path, "w", encoding="utf-8") as f:
            json.dump(split_info, f, ensure_ascii=False, indent=2)

        print(f"  Train patients: {len(train_ids)}")
        print(f"  Test patients: {len(test_ids)}")
        print(f"  划分已保存到: {split_info_path}")
    else:
        print(f"\n[INFO] 加载已有划分: {split_info_path}")
        with open(split_info_path, "r", encoding="utf-8") as f:
            split_info = json.load(f)
        train_ids = split_info["train_patient_ids"]
        test_ids = split_info["test_patient_ids"]
        print(f"  Train patients: {len(train_ids)}")
        print(f"  Test patients: {len(test_ids)}")

    base = Path(__file__).resolve().parent
    stage1_script = base / "stage1_encoder.py"
    stage2_script = base / "stage2_clinical.py"
    stage3_script = base / "stage3_transformer.py"
    stage4_script = base / "stage4_evaluate.py"

    if not args.skip_stage1:
        print("\n" + "=" * 80)
        print("Stage 1: GastroNet-5M Encoder预训练")
        print("=" * 80)
        run_script(stage1_script, config_path)
    else:
        print("\n[SKIP] Stage1 已跳过")

    stage2_summary = out_dir / "stage2" / "stage2_model_summary.json"
    if stage2_summary.exists():
        print(f"\n[SKIP] Stage2 已完成 (检测到 {stage2_summary})")
    elif not args.skip_stage2:
        print("\n" + "=" * 80)
        print("Stage 2: 临床机器学习模型 (双任务: remission + healing)")
        print("=" * 80)
        print("训练模型 (双任务: remission + healing):")
        print("  - LogisticRegression")
        print("  - RandomForest")
        print("  - XGBoost")
        print("  - SVM")
        print("  - MLP")
        print("=" * 80)
        run_script(stage2_script, config_path)
    else:
        print("\n[SKIP] Stage2 已跳过")

    stage3_summary = out_dir / "stage3" / "stage3_model_summary.json"
    if stage3_summary.exists():
        print(f"\n[SKIP] Stage3 已完成 (检测到 {stage3_summary})")
    elif not args.skip_stage3:
        print("\n" + "=" * 80)
        print("Stage 3: Transformer双任务模型 (双Head: remission + healing)")
        print("=" * 80)
        print("训练模型 (双Head多任务模式):")
        print("  - image_only (remission + healing)")
        print("  - both (remission + healing)")
        print("=" * 80)
        run_script(stage3_script, config_path)
    else:
        print("\n[SKIP] Stage3 已跳过")

    if not args.skip_stage4:
        print("\n" + "=" * 80)
        print("Stage 4: 全面模型评估与可视化（双任务: remission + healing）")
        print("=" * 80)
        print("评估内容:")
        print("  - 双任务: Remission + Healing AUC, Acc, F1, Sensitivity, Specificity")
        print("  - Bootstrap 95% CI")
        print("  - 可视化: ROC曲线, 校准曲线, 混淆矩阵, 多指标对比柱状图")
        print("  - Attention: Stage1 Encoder热力图, Attention汇总拼图")
        print("=" * 80)
        run_script(stage4_script, config_path)
    else:
        print("\n[SKIP] Stage4 已跳过")

    print("\n" + "=" * 80)
    print("Pipeline 完成!")
    print("=" * 80)
    print(f"输出目录: {out_dir}")
    print("\n生成的文件:")
    print("  - stage1/: encoder模型和权重")
    print("  - stage2/: 临床机器学习模型 (双任务: remission + healing)")
    print("    - LogisticRegression/ (pt + hf_model/)")
    print("    - RandomForest/ (pt + hf_model/)")
    print("    - XGBoost/ (pt + hf_model/)")
    print("    - SVM/ (pt + hf_model/)")
    print("    - MLP/ (pt + hf_model/)")
    print("  - stage3/: Transformer模型 (双Head架构: remission + healing)")
    print("    - image_only/ (pt + hf_model/)")
    print("    - both/ (pt + hf_model/)")
    print("  - stage4/: 评估结果和可视化")
    print("    - evaluation_results.json/csv (完整评估指标)")
    print("    - model_comparison_*.png (多模型对比图)")
    print("    - {model}_roc_*.png (ROC曲线)")
    print("    - {model}_calibration_*.png (校准曲线)")
    print("    - {model}_confusion_*.png (混淆矩阵)")
    print("    - {model}_attention_summary.png (Attention汇总)")
    print("    - attention_encoder/ (Stage1 Encoder热力图)")

if __name__ == "__main__":
    main()
