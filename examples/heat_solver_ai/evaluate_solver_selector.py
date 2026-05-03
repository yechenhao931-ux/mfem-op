#!/usr/bin/env python3
"""
evaluate_solver_selector.py
===========================
对训练好的 solver_selector.pt 做端到端评估：

  * Top-1 / Top-3 准确率
  * 期望耗时倍率（regret）：predicted_time / oracle_time
        - mean / max / 90 分位
        - 越接近 1.0 越好
  * 按 (problem, dim) 切片的胜负分布
  * 失败案例 Top-N 打印（实际跟最优差距最大的几条）

CSV 必须是 heat_bench 的输出（含 converged 列）。可与训练同源（重新评估
训练集）也可独立留出测试集。

用法：
    python3 evaluate_solver_selector.py \
        --csv heat_solver_data.csv \
        --model solver_selector.pt
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# 复用训练脚本里的常量与特征工程
sys.path.insert(0, os.path.dirname(__file__))
from train_solver_selector import (   # noqa: E402
    PROBLEMS, MESH_TYPES, SOLVERS, NUMERIC_FEATURES,
    engineer_features, one_hot,
)


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    feats   = engineer_features(df).to_numpy(dtype=np.float32)
    prob_oh = one_hot(df["problem"],   PROBLEMS)
    mesh_oh = one_hot(df["mesh_type"], MESH_TYPES)
    return np.concatenate([feats, prob_oh, mesh_oh], axis=1).astype(np.float32)


def evaluate(args):
    df = pd.read_csv(args.csv)
    df = df[df["converged"] == 1].copy()
    if df.empty:
        raise RuntimeError("CSV 中无收敛样本。")

    keys = ["problem", "mesh_type", "dim", "poly_order", "ref_level"]
    df["solver_id"] = df["solver"].apply(
        lambda s: SOLVERS.index(s) if s in SOLVERS else -1)
    df = df[df["solver_id"] >= 0]

    # 每个 (group, solver) 的 total_ms。某些组合某些求解器没收敛就为 NaN
    pivot = df.pivot_table(
        index=keys, columns="solver", values="total_ms", aggfunc="min")
    # 拿每组任一行做特征（特征对同一组里不同求解器是相同的）
    rep = df.drop_duplicates(keys).set_index(keys)
    X = build_feature_matrix(rep)

    # 推理
    model = torch.jit.load(args.model)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X)).cpu().numpy()
    # 按降序排，得到 ranking
    ranking = np.argsort(-logits, axis=1)

    # 标签 = 该组耗时最短的（已收敛的）求解器 id
    times = pivot.reindex(columns=SOLVERS).to_numpy(dtype=np.float64)
    valid_mask = ~np.isnan(times)

    # oracle = 每行最小耗时
    oracle_idx = np.array([
        np.nanargmin(row) if np.any(~np.isnan(row)) else -1
        for row in times])
    oracle_time = np.array([
        np.nanmin(row) if np.any(~np.isnan(row)) else np.nan
        for row in times])

    n = X.shape[0]
    correct_top1 = 0
    correct_top3 = 0
    regrets = []
    actual_solvers = []
    pred_solvers   = []
    per_problem = {p: {"n": 0, "top1": 0, "regret_sum": 0.0}
                   for p in PROBLEMS}

    for i in range(n):
        if oracle_idx[i] < 0:
            continue
        pred_id = ranking[i, 0]

        # 若预测求解器在该样本上没收敛，回退到下一名 —— 模拟真实使用场景
        chosen_id = pred_id
        for j in range(ranking.shape[1]):
            cand = ranking[i, j]
            if valid_mask[i, cand]:
                chosen_id = cand
                break

        actual_solvers.append(SOLVERS[oracle_idx[i]])
        pred_solvers.append(SOLVERS[chosen_id])

        if chosen_id == oracle_idx[i]:
            correct_top1 += 1
        if oracle_idx[i] in ranking[i, :3]:
            correct_top3 += 1

        # regret = 选择耗时 / 最优耗时
        chosen_time = times[i, chosen_id]
        if not np.isnan(chosen_time) and oracle_time[i] > 0:
            regrets.append(chosen_time / oracle_time[i])

        problem = rep.index[i][0]
        per_problem[problem]["n"]    += 1
        per_problem[problem]["top1"] += int(chosen_id == oracle_idx[i])
        if not np.isnan(chosen_time):
            per_problem[problem]["regret_sum"] += chosen_time / oracle_time[i]

    # ----------------------------------------------------------------------
    # 报告
    # ----------------------------------------------------------------------
    print(f"\n=== 评估 {args.csv} | 模型 {args.model} ===")
    print(f"  样本组数: {n}")
    print(f"  Top-1 准确率: {correct_top1 / max(n,1):.3f}")
    print(f"  Top-3 准确率: {correct_top3 / max(n,1):.3f}")
    if regrets:
        regrets_arr = np.array(regrets)
        print(f"  Regret (chosen/oracle):")
        print(f"      mean = {regrets_arr.mean():.3f}")
        print(f"      p50  = {np.percentile(regrets_arr, 50):.3f}")
        print(f"      p90  = {np.percentile(regrets_arr, 90):.3f}")
        print(f"      max  = {regrets_arr.max():.3f}")
    print(f"\n  按问题类型:")
    print(f"  {'problem':<10} {'N':>5} {'top1':>7} {'regret':>8}")
    for p in PROBLEMS:
        r = per_problem[p]
        if r["n"] == 0:
            continue
        avg_regret = r["regret_sum"] / r["n"]
        print(f"  {p:<10} {r['n']:>5d} {r['top1']/r['n']:>7.3f} "
              f"{avg_regret:>8.3f}")

    # 失败案例
    if regrets:
        worst_idx = np.argsort(-np.array(regrets))[:args.show_worst]
        print(f"\n  最差 {len(worst_idx)} 个预测：")
        # 注意 regrets 与 rep.index 的对应关系
        kept = [i for i in range(n) if oracle_idx[i] >= 0
                and not np.isnan(times[i, ranking[i, 0]])]
        for k in worst_idx:
            if k >= len(kept): continue
            i = kept[k]
            key = rep.index[i]
            pid = ranking[i, 0]
            print(f"    {key} → 选 {SOLVERS[pid]:<14} "
                  f"({times[i, pid]:>7.2f} ms) "
                  f"oracle={SOLVERS[oracle_idx[i]]:<14} "
                  f"({oracle_time[i]:>7.2f} ms) "
                  f"× {regrets[k]:.2f}")

    if args.out_csv:
        out = pd.DataFrame({
            "actual": actual_solvers,
            "predicted": pred_solvers,
        })
        out.to_csv(args.out_csv, index=False)
        print(f"\n  → 预测细节写入 {args.out_csv}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",         required=True)
    p.add_argument("--model",       default=str(
        Path(__file__).with_name("solver_selector.pt")))
    p.add_argument("--show-worst",  type=int, default=10)
    p.add_argument("--out-csv",     default="")
    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
