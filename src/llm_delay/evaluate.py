from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ─────────────────────────────────────────────
# Guard yardımcıları
# ─────────────────────────────────────────────

def _safe_pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Pearson korelasyonu — sabit dizi veya tek sample durumunda nan döner.
    scipy.stats.pearsonr sıfır varyans veya n<2 durumunda exception fırlatır.
    """
    if len(x) < 2 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return float("nan"), float("nan")
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Spearman korelasyonu — sabit dizi veya tek sample durumunda nan döner.
    """
    if len(x) < 2 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return float("nan"), float("nan")
    r, p = stats.spearmanr(x, y)
    return float(r), float(p)


def _safe_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Concordance Correlation Coefficient — hem precision hem accuracy ölçer.
    Sıfır varyans durumunda nan döner.
    """
    var_t = np.var(y_true)
    var_p = np.var(y_pred)
    if var_t < 1e-10 and var_p < 1e-10:
        return float("nan")
    mean_t = np.mean(y_true)
    mean_p = np.mean(y_pred)
    cov    = np.mean((y_true - mean_t) * (y_pred - mean_p))
    return float(2 * cov / (var_t + var_p + (mean_t - mean_p) ** 2 + 1e-12))


# ─────────────────────────────────────────────
# Core metrics
# ─────────────────────────────────────────────

def regression_metrics(
    y_true: List[float] | np.ndarray,
    y_pred: List[float] | np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    residuals = y_true - y_pred
    abs_res   = np.abs(residuals)
    n         = len(y_true)

    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mse  = float(mean_squared_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))

    mask = y_true != 0
    mape = float(np.mean(np.abs(residuals[mask] / y_true[mask])) * 100) if mask.any() else float("nan")

    mae_p50 = float(np.percentile(abs_res, 50))
    mae_p90 = float(np.percentile(abs_res, 90))
    mae_p95 = float(np.percentile(abs_res, 95))
    mae_p99 = float(np.percentile(abs_res, 99))
    max_err = float(np.max(abs_res))

    mean_bias = float(np.mean(residuals))    
    std_res   = float(np.std(residuals))

    tstat, tpval = stats.ttest_1samp(residuals, 0.0)
    systematic_bias = bool(tpval < 0.05)

    # ── Korelasyon ───────────────────────────
    pearson_r,  pearson_p  = _safe_pearson(y_true, y_pred)
    spearman_r, spearman_p = _safe_spearman(y_true, y_pred)
    ccc = _safe_ccc(y_true, y_pred)

    # ── Normallik testi (residuals) ──────────
    # Shapiro-Wilk ≤5000 sample; üstünde Kolmogorov-Smirnov
    if n <= 5000:
        normality_stat, normality_p = stats.shapiro(residuals)
        normality_test = "shapiro"
    else:
        normality_stat, normality_p = stats.kstest(
            residuals, "norm",
            args=(float(np.mean(residuals)), float(np.std(residuals))),
        )
        normality_test = "ks"

    residuals_normal = bool(float(normality_p) > 0.05)

    # ── Skewness & Kurtosis (residuals) ─────
    skewness = float(stats.skew(residuals))
    kurtosis = float(stats.kurtosis(residuals))   # excess kurtosis

    # ── Within-threshold accuracy ────────────
    within_1  = float(np.mean(abs_res <= 1.0)  * 100)
    within_2  = float(np.mean(abs_res <= 2.0)  * 100)
    within_5  = float(np.mean(abs_res <= 5.0)  * 100)
    within_10 = float(np.mean(abs_res <= 10.0) * 100)

    # ── Sonuç dict ───────────────────────────
    p = prefix + "_" if prefix else ""

    return {
        # Temel
        f"{p}mae":   mae,
        f"{p}rmse":  rmse,
        f"{p}mse":   mse,
        f"{p}r2":    r2,
        f"{p}mape":  mape,

        # Hata dağılımı
        f"{p}mae_p50":  mae_p50,
        f"{p}mae_p90":  mae_p90,
        f"{p}mae_p95":  mae_p95,
        f"{p}mae_p99":  mae_p99,
        f"{p}max_err":  max_err,

        # Bias
        f"{p}mean_bias":       mean_bias,
        f"{p}std_residuals":   std_res,
        f"{p}systematic_bias": systematic_bias,
        f"{p}bias_tstat":      float(tstat),
        f"{p}bias_tpval":      float(tpval),

        # Korelasyon
        f"{p}pearson_r":   pearson_r,
        f"{p}pearson_p":   pearson_p,
        f"{p}spearman_r":  spearman_r,
        f"{p}spearman_p":  spearman_p,
        f"{p}ccc":         ccc,

        # Residual istatistikleri
        f"{p}residual_skewness": skewness,
        f"{p}residual_kurtosis": kurtosis,
        f"{p}normality_test":    normality_test,
        f"{p}normality_stat":    float(normality_stat),
        f"{p}normality_p":       float(normality_p),
        f"{p}residuals_normal":  residuals_normal,

        # Within-threshold
        f"{p}within_1min_pct":  within_1,
        f"{p}within_2min_pct":  within_2,
        f"{p}within_5min_pct":  within_5,
        f"{p}within_10min_pct": within_10,

        # Meta
        f"{p}n_samples": int(n),
    }



def print_metrics(metrics: Dict[str, float], title: str = "Metrics") -> None:
    """Metrikleri okunabilir formatta yazdır."""
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")

    sections = {
        "Temel Hata":          ["mae", "rmse", "mse", "mape"],
        "R² / Korelasyon":     ["r2", "pearson_r", "spearman_r", "ccc"],
        "Hata Dağılımı (abs)": ["mae_p50", "mae_p90", "mae_p95", "mae_p99", "max_err"],
        "Bias Analizi":        ["mean_bias", "std_residuals", "systematic_bias", "bias_tpval"],
        "Residual İstatistik": ["residual_skewness", "residual_kurtosis", "residuals_normal"],
        "±N Dakika İçinde (%)":["within_1min_pct", "within_2min_pct", "within_5min_pct", "within_10min_pct"],
    }

    # prefix varsa bul
    sample_key = next(iter(metrics))
    prefix = ""
    for s in ["train_", "val_", "test_"]:
        if sample_key.startswith(s):
            prefix = s
            break

    for section, keys in sections.items():
        print(f"\n  [{section}]")
        for k in keys:
            val = metrics.get(f"{prefix}{k}")
            if val is None:
                continue
            if isinstance(val, bool):
                print(f"    {k:<30s}: {val}")
            elif isinstance(val, float):
                print(f"    {k:<30s}: {val:.4f}")
            else:
                print(f"    {k:<30s}: {val}")

    print(f"\n  n_samples: {metrics.get(f'{prefix}n_samples', '?')}")
    print(f"{'=' * 55}\n")


# ─────────────────────────────────────────────
# Train / val / test karşılaştırma
# ─────────────────────────────────────────────

def compare_splits(
    train_true, train_pred,
    val_true,   val_pred,
    test_true,  test_pred,
) -> Dict[str, Dict]:
   
    train_m = regression_metrics(train_true, train_pred, prefix="train")
    val_m   = regression_metrics(val_true,   val_pred,   prefix="val")
    test_m  = regression_metrics(test_true,  test_pred,  prefix="test")

    mae_gap_tv = train_m["train_mae"] - val_m["val_mae"]
    mae_gap_vt = val_m["val_mae"]     - test_m["test_mae"]
    r2_drop    = train_m["train_r2"]  - test_m["test_r2"]

    overfit_flag = bool(r2_drop > 0.05 or abs(mae_gap_tv) > 1.0)

    summary = {
        "train_mae":  train_m["train_mae"],
        "val_mae":    val_m["val_mae"],
        "test_mae":   test_m["test_mae"],
        "train_r2":   train_m["train_r2"],
        "val_r2":     val_m["val_r2"],
        "test_r2":    test_m["test_r2"],
        "train_rmse": train_m["train_rmse"],
        "val_rmse":   val_m["val_rmse"],
        "test_rmse":  test_m["test_rmse"],
        "mae_gap_train_val":  float(mae_gap_tv),
        "mae_gap_val_test":   float(mae_gap_vt),
        "r2_drop_train_test": float(r2_drop),
        "overfit_flag":       overfit_flag,
        "test_within_2min_pct": test_m["test_within_2min_pct"],
        "test_within_5min_pct": test_m["test_within_5min_pct"],
        "test_systematic_bias": test_m["test_systematic_bias"],
        "test_ccc":             test_m["test_ccc"],
        "test_spearman_r":      test_m["test_spearman_r"],
    }

    return {
        "train":   train_m,
        "val":     val_m,
        "test":    test_m,
        "summary": summary,
    }