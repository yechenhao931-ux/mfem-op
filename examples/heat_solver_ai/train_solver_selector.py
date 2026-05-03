#!/usr/bin/env python3
"""
train_solver_selector.py
========================
读取 heat_bench 产出的 CSV，训练一个 MLP 分类器：
    输入：问题 + 网格 + 矩阵特征
    输出：在该问题上耗时最短的求解器类别

用法：
    python3 train_solver_selector.py --csv ../heat_solver_data.csv

产物（与脚本同目录）：
    solver_selector.pt        TorchScript 模型，C++ 通过 LibTorch 加载
    solver_selector_meta.json 类别映射、特征顺序、标准化参数
    training_report.txt       验证集准确率、各类预测分布
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------------------------------------------------
# 类别（必须与 C++ 推理端、benchmark 保持一致）
# ----------------------------------------------------------------------
PROBLEMS   = ["iso", "aniso", "var", "react", "trans", "conv"]
MESH_TYPES = ["quad", "tri", "hex", "tet"]
SOLVERS    = ["CG", "PCG_Jacobi", "PCG_GS", "PCG_Cheby",
              "MINRES", "GMRES", "GMRES_Jacobi",
              "BiCGSTAB", "BiCGSTAB_Jacobi", "DIRECT_UMF"]

# 数值特征（顺序固定，C++ 端按此顺序构造输入向量）
NUMERIC_FEATURES = [
    "dim", "poly_order",
    "log_n_dof", "log_n_elements", "log_nnz",
    "nnz_per_row", "sparsity", "symmetry_ratio", "diag_dominance",
    "is_spd", "log_frob_norm",
    "log_kappa", "log_aniso", "reaction_coef",
    "log_dt", "log_peclet",
]


def safe_log10(x, eps=1e-12):
    return np.log10(np.maximum(np.asarray(x, dtype=np.float64), eps))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """从原始 CSV 列生成特征列。"""
    f = pd.DataFrame()
    f["dim"]            = df["dim"].astype(float)
    f["poly_order"]     = df["poly_order"].astype(float)
    f["log_n_dof"]      = safe_log10(df["n_dof"])
    f["log_n_elements"] = safe_log10(df["n_elements"])
    f["log_nnz"]        = safe_log10(df["nnz"])
    f["nnz_per_row"]    = df["nnz_per_row"].astype(float)
    f["sparsity"]       = df["sparsity"].astype(float)
    f["symmetry_ratio"] = df["symmetry_ratio"].astype(float)
    f["diag_dominance"] = np.minimum(df["diag_dominance"].astype(float), 100.0)
    f["is_spd"]         = df["is_spd"].astype(float)
    f["log_frob_norm"]  = safe_log10(df["frob_norm_est"])
    f["log_kappa"]      = safe_log10(df["kappa_iso"])
    f["log_aniso"]      = safe_log10(df["aniso_ratio"])
    f["reaction_coef"]  = df["reaction_coef"].astype(float)
    f["log_dt"]         = safe_log10(df["dt_step"] + 1e-12)
    f["log_peclet"]     = safe_log10(df["peclet"] + 1e-12)
    return f[NUMERIC_FEATURES]


def one_hot(idx_series: pd.Series, vocab: list) -> np.ndarray:
    table = {v: i for i, v in enumerate(vocab)}
    out = np.zeros((len(idx_series), len(vocab)), dtype=np.float32)
    for i, v in enumerate(idx_series):
        if v in table:
            out[i, table[v]] = 1.0
    return out


def build_training_set(df: pd.DataFrame):
    """对每个 (problem, mesh_type, dim, poly_order, ref_level) 组，
    取收敛求解器中 total_ms 最小者作为标签。"""
    df = df[df["converged"] == 1].copy()
    if df.empty:
        raise RuntimeError("CSV 里没有收敛样本，无法训练。")

    df["solver_id"] = df["solver"].apply(
        lambda s: SOLVERS.index(s) if s in SOLVERS else -1)
    df = df[df["solver_id"] >= 0]

    keys = ["problem", "mesh_type", "dim", "poly_order", "ref_level"]
    best = df.loc[df.groupby(keys)["total_ms"].idxmin()].reset_index(drop=True)
    print(f"  {len(df)} 个收敛样本 → {len(best)} 个 (问题, 配置) 组")

    feats   = engineer_features(best).to_numpy(dtype=np.float32)
    prob_oh = one_hot(best["problem"],   PROBLEMS)
    mesh_oh = one_hot(best["mesh_type"], MESH_TYPES)
    X = np.concatenate([feats, prob_oh, mesh_oh], axis=1).astype(np.float32)
    y = best["solver_id"].to_numpy(dtype=np.int64)
    return X, y, best


# ----------------------------------------------------------------------
# 模型：含特征标准化的端到端 MLP（标准化嵌入到模型，C++ 端无需重复实现）
# ----------------------------------------------------------------------
class SolverSelector(nn.Module):
    def __init__(self, n_input: int, n_classes: int,
                 mean: torch.Tensor, std: torch.Tensor,
                 hidden: int = 96):
        super().__init__()
        self.register_buffer("feat_mean", mean)
        self.register_buffer("feat_std",  std)
        self.net = nn.Sequential(
            nn.Linear(n_input, hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.feat_mean) / self.feat_std
        return self.net(x)


def train(args):
    df = pd.read_csv(args.csv)
    if df.empty:
        raise RuntimeError(f"{args.csv} 为空")
    print(f"读取 {len(df)} 行数据，列: {list(df.columns)[:8]}...")

    X, y, best_df = build_training_set(df)
    print(f"特征维度: {X.shape}, 标签分布: "
          f"{dict(zip(*np.unique(y, return_counts=True)))}")

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(X))
    n_val = max(int(len(X) * 0.2), 1)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    # 数值列单独标准化；one-hot 列保持原值
    n_numeric = len(NUMERIC_FEATURES)
    mean = np.zeros(X.shape[1], dtype=np.float32)
    std  = np.ones(X.shape[1],  dtype=np.float32)
    mean[:n_numeric] = X[train_idx, :n_numeric].mean(axis=0)
    std[:n_numeric]  = X[train_idx, :n_numeric].std(axis=0) + 1e-6

    n_classes = len(SOLVERS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SolverSelector(
        n_input=X.shape[1], n_classes=n_classes,
        mean=torch.tensor(mean), std=torch.tensor(std),
        hidden=args.hidden,
    ).to(device)

    Xt = torch.tensor(X[train_idx]); yt = torch.tensor(y[train_idx])
    Xv = torch.tensor(X[val_idx]);   yv = torch.tensor(y[val_idx])
    train_loader = DataLoader(TensorDataset(Xt, yt),
                              batch_size=args.batch_size, shuffle=True)

    # 类别加权 → 缓解长尾 (有些求解器极少胜出)
    class_counts = np.bincount(y[train_idx], minlength=n_classes).astype(
        np.float32)
    class_weights = 1.0 / np.maximum(class_counts, 1.0)
    class_weights = class_weights * (n_classes / class_weights.sum())
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss   = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        train_loss = running / max(len(train_idx), 1)

        model.eval()
        with torch.no_grad():
            val_logits = model(Xv.to(device)).cpu().numpy()
        preds   = val_logits.argmax(axis=1)
        topk    = np.argsort(-val_logits, axis=1)[:, :3]
        val_acc = float((preds == y[val_idx]).mean())
        val_top3 = float(np.mean([y[val_idx][i] in topk[i]
                                  for i in range(len(val_idx))]))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}

        if (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}/{args.epochs}  "
                  f"loss={train_loss:.4f}  "
                  f"val_acc={val_acc:.3f}  top3={val_top3:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    # 重新跑一次最佳模型在 val 上以拿到最终预测
    model.eval()
    with torch.no_grad():
        final_val_logits = model(Xv.to(device)).cpu().numpy()
    preds = final_val_logits.argmax(axis=1)
    topk  = np.argsort(-final_val_logits, axis=1)[:, :3]
    final_top3 = float(np.mean([y[val_idx][i] in topk[i]
                                for i in range(len(val_idx))]))
    print(f"\n最佳验证 Top-1: {best_val_acc:.3f}  Top-3: {final_top3:.3f}")

    # ------------------------------------------------------------------
    # 导出 TorchScript（C++ 推理端通过 torch::jit::load 加载）
    # ------------------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval().cpu()
    example = torch.zeros(1, X.shape[1], dtype=torch.float32)
    scripted = torch.jit.trace(model, example)
    pt_path = out_dir / "solver_selector.pt"
    scripted.save(str(pt_path))
    print(f"  TorchScript → {pt_path}")

    meta = {
        "problems":         PROBLEMS,
        "mesh_types":       MESH_TYPES,
        "solvers":          SOLVERS,
        "numeric_features": NUMERIC_FEATURES,
        "feature_dim":      int(X.shape[1]),
        "n_classes":        n_classes,
        "val_accuracy":     float(best_val_acc),
        "val_top3":         float(final_top3),
        "training_size":    int(len(train_idx)),
        "val_size":         int(n_val),
    }
    json_path = out_dir / "solver_selector_meta.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  Meta JSON   → {json_path}")

    # ------------------------------------------------------------------
    # 报告：验证集上每个真实标签的预测分布
    # ------------------------------------------------------------------
    report = [f"val_top1 = {best_val_acc:.4f}",
              f"val_top3 = {final_top3:.4f}",
              f"n_train = {len(train_idx)}, n_val = {n_val}\n"]
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y[val_idx], preds):
        cm[true, pred] += 1
    report.append("混淆矩阵 (行=真实, 列=预测):")
    header = "       " + "  ".join(f"{s[:8]:>8}" for s in SOLVERS)
    report.append(header)
    for i, name in enumerate(SOLVERS):
        row = "  ".join(f"{cm[i, j]:>8d}" for j in range(n_classes))
        report.append(f"{name[:7]:>7}  {row}")
    report_path = out_dir / "training_report.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"  Report      → {report_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="heat_bench 输出的 CSV")
    p.add_argument("--out-dir", default=os.path.dirname(__file__) or ".",
                   help="输出目录")
    p.add_argument("--epochs",     type=int,   default=200)
    p.add_argument("--batch-size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--hidden",     type=int,   default=96)
    p.add_argument("--seed",       type=int,   default=42)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
