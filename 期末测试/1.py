# mc_porous_inversion.py
# -*- coding: utf-8 -*-
"""
Monte Carlo 模拟数据驱动的多孔介质参数反演建模（完整可运行示例）
- 前向（物理）模型：Darcy + Kozeny–Carman（用于合成数据）
- 反演（统计）模型：回归预测 log10(k)
- 对比模型：Linear / Lasso / SVR(RBF) / RandomForest
- 输出：metrics_summary.csv、figures/*.png、predictions_*.csv

依赖：
pip install numpy pandas scikit-learn matplotlib

运行：
python mc_porous_inversion.py
"""

from __future__ import annotations

import os
import math
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Dict, Tuple, List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# -----------------------------
# 0) 工具函数
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)


def rmse(y_true, y_pred) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


# -----------------------------
# 1) 理论基础（用于合成数据的“前向模型”）
# -----------------------------
@dataclass
class PhysicalConfig:
    """物理/采样设置（你可以按报告需要改这里）"""
    mu: float = 1.0e-3               # 动力黏度 Pa·s（示例取水量级）
    Ck: float = 5.0                  # Kozeny 常数（量级示意）
    dpdx_range: Tuple[float, float] = (1e2, 5e4)  # 压力梯度 |dP/dx|，Pa/m
    phi_range: Tuple[float, float] = (0.10, 0.45) # 孔隙率范围
    S_range: Tuple[float, float] = (5e4, 2e5)     # 比表面积 1/m（示意量级）
    obs_noise_std: float = 0.03      # 观测噪声：相对噪声（给 v_obs）


def kozeny_carman_k(phi: np.ndarray, S: np.ndarray, Ck: float) -> np.ndarray:
    """
    Kozeny–Carman: k = (phi^3 / (Ck*(1-phi)^2)) * (1/S^2)
    说明：这里只是合成数据的“合理物理生成器”，不是唯一真理。
    """
    eps = 1e-12
    phi = np.clip(phi, eps, 1 - eps)
    return (phi**3 / (Ck * (1 - phi)**2)) * (1.0 / (S**2))


def darcy_velocity(k: np.ndarray, mu: float, dpdx: np.ndarray) -> np.ndarray:
    """
    Darcy: v = -(k/mu) * dP/dx
    这里只使用 1D 形式，符号不重要（反演通常用幅值或含符号观测皆可）
    """
    return -(k / mu) * dpdx


def monte_carlo_generate(N: int, cfg: PhysicalConfig, seed: int = 42) -> pd.DataFrame:
    """
    Monte Carlo 生成合成数据：
    - 随机采样 phi、S、dP/dx
    - 计算 k_true, v_true
    - 加观测噪声得到 v_obs
    - 目标：y = log10(k_true)
    - 特征：phi, |dpdx|, v_obs（可自行扩展）
    """
    set_global_seed(seed)

    phi = np.random.uniform(cfg.phi_range[0], cfg.phi_range[1], size=N)
    S = np.random.uniform(cfg.S_range[0], cfg.S_range[1], size=N)
    dpdx = np.random.uniform(cfg.dpdx_range[0], cfg.dpdx_range[1], size=N)
    sign = np.random.choice([-1.0, 1.0], size=N)  # 允许梯度有方向
    dpdx = dpdx * sign

    k_true = kozeny_carman_k(phi, S, cfg.Ck)
    v_true = darcy_velocity(k_true, cfg.mu, dpdx)

    # 相对噪声：v_obs = v_true * (1 + noise)
    noise = np.random.normal(loc=0.0, scale=cfg.obs_noise_std, size=N)
    v_obs = v_true * (1.0 + noise)

    df = pd.DataFrame({
        "phi": phi,
        "S": S,
        "dpdx": dpdx,
        "abs_dpdx": np.abs(dpdx),
        "k_true": k_true,
        "log10_k": np.log10(k_true),
        "v_true": v_true,
        "v_obs": v_obs,
    })
    return df


# -----------------------------
# 2) 回归建模（统计意义）
# -----------------------------
def build_models(random_state: int = 0) -> Dict[str, object]:
    """
    返回要对比的模型：
    - 线性模型：解释性强，但对非线性/交互不敏感
    - Lasso：带L1正则，做稀疏化（特征多时更有用）
    - SVR(RBF)：非线性核回归，适合中小规模
    - RandomForest：非参数集成，强非线性拟合能力，解释性相对弱
    """
    models = {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "Lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.01, max_iter=20000, random_state=random_state))
        ]),
        "SVR_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=30.0, gamma="scale", epsilon=0.02))
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
    }
    return models


def evaluate_once(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    test_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    单次划分训练/测试并评估所有模型。
    训练目标：log10(k)
    评价空间：k
    返回：
    - metrics_df: 各模型的 MSE/MAE/R2（在 k 空间）
    - preds: 每个模型的 (k_true, k_pred)（用于散点图/保存）
    """
    X = df[feature_cols].values
    y_log = df[target_col].values  # 这里是 log10_k

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=test_size, random_state=seed
    )

    models = build_models(random_state=seed)

    rows = []
    preds = {}
    for name, model in models.items():
        model.fit(X_train, y_train_log)
        y_pred_log = model.predict(X_test)

        # 反变换到 k 空间
        k_true = 10 ** y_test_log
        k_pred = 10 ** y_pred_log

        rows.append({
            "Model": name,
            "MSE": mean_squared_error(k_true, k_pred),
            "MAE": mean_absolute_error(k_true, k_pred),
            "R2": r2_score(k_true, k_pred),
        })
        preds[name] = (k_true.copy(), k_pred.copy())

    metrics_df = pd.DataFrame(rows).sort_values("MSE")
    return metrics_df, preds



def evaluate_stability(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "log10_k",
    test_size: float = 0.2,
    repeats: int = 10,
    base_seed: int = 123,
) -> pd.DataFrame:
    all_metrics = []
    for i in range(repeats):
        seed = base_seed + i
        metrics_i, _ = evaluate_once(df, feature_cols, target_col, test_size, seed)
        metrics_i["repeat"] = i
        all_metrics.append(metrics_i)

    big = pd.concat(all_metrics, ignore_index=True)
    summary = (big.groupby("Model")[["MSE", "MAE", "R2"]]
               .agg(["mean", "std"])
               .reset_index())
    return summary



# -----------------------------
# 3) 画图（报告用）
# -----------------------------
def plot_porous_media_schematic(outpath: str) -> None:
    """
    多孔介质参数示意图（自绘：固体颗粒+孔隙+流动方向）
    你可以把它放在第二节“理论基础”里。
    """
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.set_title("Schematic of Porous Media (solids / pores / flow)")

    # 画一堆圆形当固体颗粒
    rng = np.random.default_rng(0)
    for _ in range(35):
        x, y = rng.uniform(0.05, 0.95), rng.uniform(0.10, 0.90)
        r = rng.uniform(0.03, 0.06)
        circle = plt.Circle((x, y), r, fill=True, alpha=0.35)
        ax.add_patch(circle)

    # 流动箭头
    ax.annotate("", xy=(0.95, 0.05), xytext=(0.05, 0.05),
                arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(0.50, 0.01, "Flow direction", ha="center", va="bottom")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_data_distributions(df: pd.DataFrame, outdir: str) -> None:
    """
    数据分布图：phi、abs_dpdx、v_obs、log10_k
    """
    cols = ["phi", "abs_dpdx", "v_obs", "log10_k"]
    for c in cols:
        fig = plt.figure(figsize=(5.5, 4))
        ax = fig.add_subplot(111)
        ax.hist(df[c].values, bins=30)
        ax.set_title(f"Distribution of {c}")
        ax.set_xlabel(c)
        ax.set_ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"dist_{c}.png"), dpi=200)
        plt.close(fig)


def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, title: str, outpath: str) -> None:
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.scatter(y_true, y_pred, s=12, alpha=0.7)

    # 关键：k 跨数量级，建议对数坐标
    ax.set_xscale("log")
    ax.set_yscale("log")

    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], lw=2)

    ax.set_title(title)
    ax.set_xlabel("True k")
    ax.set_ylabel("Predicted k")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)



def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, title: str, outpath: str) -> None:
    resid = y_pred - y_true
    fig = plt.figure(figsize=(5.5, 4))
    ax = fig.add_subplot(111)
    ax.scatter(y_true, resid, s=12, alpha=0.7)
    ax.axhline(0, lw=2)
    ax.set_title(title)
    ax.set_xlabel("True")
    ax.set_ylabel("Residual (Pred-True)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_sample_size_effect(cfg: PhysicalConfig, feature_cols: List[str], outpath: str) -> None:
    """
    展示 Monte Carlo 样本量 N 对误差的影响（以 SVR / RF 为例）
    """
    Ns = [80, 120, 200, 300, 500, 800]
    records = []

    for N in Ns:
        df = monte_carlo_generate(N=N, cfg=cfg, seed=42)
        metrics, _ = evaluate_once(df, feature_cols, "log10_k", test_size=0.2, seed=7)
        # 取两种模型
        for m in ["SVR_RBF", "RandomForest"]:
            row = metrics[metrics["Model"] == m].iloc[0].to_dict()
            records.append({"N": N, **row})

    dfm = pd.DataFrame(records)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    for m in ["SVR_RBF", "RandomForest"]:
        sub = dfm[dfm["Model"] == m].sort_values("N")
        ax.plot(sub["N"], sub["MSE"], marker="o", label=m)

    ax.set_title("Effect of Monte Carlo sample size on MSE")
    ax.set_xlabel("N (samples)")
    ax.set_ylabel("MSE")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)


# -----------------------------
# 4) 主流程
# -----------------------------
def main():
    outdir = "outputs"
    figdir = os.path.join(outdir, "figures")
    ensure_dir(outdir)
    ensure_dir(figdir)

    # 物理&噪声配置（你可以在报告“方法与数据”说明这些范围/假设）
    cfg = PhysicalConfig(
        mu=1.0e-3,
        Ck=5.0,
        dpdx_range=(1e2, 5e4),
        phi_range=(0.10, 0.45),
        S_range=(5e4, 2e5),
        obs_noise_std=0.03,
    )

    # 生成 Monte Carlo 合成数据
    N = 500
    df = monte_carlo_generate(N=N, cfg=cfg, seed=42)

    # 选择特征（你也可以把 S 当作“不可观测”然后不放入特征）
    feature_cols = ["phi", "abs_dpdx", "v_obs"]  # 常见设定：phi、压梯、观测速度
    target_col = "log10_k"

    # 画“多孔介质示意图”和数据分布图
    plot_porous_media_schematic(os.path.join(figdir, "porous_media_schematic.png"))
    plot_data_distributions(df, figdir)

    # 单次评估 + 输出预测图
    metrics_df, preds = evaluate_once(df, feature_cols, target_col, test_size=0.2, seed=7)
    metrics_df.to_csv(os.path.join(outdir, "metrics_once.csv"), index=False)

    # 保存每个模型的 True vs Pred / Residuals 图 & 预测CSV
    for name, (y_true, y_pred) in preds.items():
        plot_pred_vs_true(
            y_true, y_pred,
            title=f"True vs Predicted ({name})",
            outpath=os.path.join(figdir, f"pred_vs_true_{name}.png")
        )
        plot_residuals(
            y_true, y_pred,
            title=f"Residual Plot ({name})",
            outpath=os.path.join(figdir, f"residuals_{name}.png")
        )
        pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(
            os.path.join(outdir, f"predictions_{name}.csv"), index=False
        )

    # 稳定性评估（重复划分）
    summary = evaluate_stability(df, feature_cols, target_col, test_size=0.2, repeats=10, base_seed=100)
    summary.to_csv(os.path.join(outdir, "metrics_summary.csv"), index=False)

    # 样本量影响曲线
    plot_sample_size_effect(cfg, feature_cols, os.path.join(figdir, "rmse_vs_N.png"))

    # 保存配置，方便报告复现
    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, ensure_ascii=False, indent=2)

    # 终端打印要点
    print("=== Single split metrics (sorted by RMSE) ===")
    print(metrics_df.to_string(index=False))
    print("\n=== Stability summary (mean±std over repeats) saved to outputs/metrics_summary.csv ===")
    print("Done. See outputs/ and outputs/figures/.")


if __name__ == "__main__":
    main()
