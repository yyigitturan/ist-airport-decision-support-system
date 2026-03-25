from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


# ══════════════════════════════════════════════════════════════════════
# Sabitler
# ══════════════════════════════════════════════════════════════════════

_EPS: float = 1e-9          # tüm division-by-zero koruma noktaları
_ALPHA: float = 0.05        # istatistiksel testlerde α (95 % güven)

_SHAPIRO_MAX_N: int   = 5_000
_ANDERSON_MAX_N: int  = 25_000
_MIN_CORR_N: int      = 3

# Heuristic eşikler — flag üretimi
_BIAS_EFFECT_THRESH: float       = 0.2   # |mean_bias|/std_res
_HEAVY_TAIL_KURTOSIS: float      = 1.0   # excess kurtosis
_HEAVY_TAIL_P99_P50_RATIO: float = 4.0   # P99/P50 oranı (kurtosis tek başına yetersiz)
_NONLINEAR_SPEARMAN_MIN: float   = 0.6   # Spearman bu değerin üzerindeyken…
_NONLINEAR_PEARSON_GAP: float    = 0.15  # …Pearson bu kadar düşükse → nonlinear
_UNSTABLE_CORR_GAP: float        = 0.10  # |pearson - spearman| farkı
_UNSTABLE_CORR_ABS: float        = 0.5   # pearson mutlak değeri bu altındaysa + gap varsa
_CALIB_SLOPE_TOL: float          = 0.10  # |slope - 1| toleransı
_CALIB_INTERCEPT_RATIO: float    = 0.25  # |intercept| / std_res oranı
_HETERO_SPEARMAN_P: float        = 0.05  # heteroscedasticity Spearman p eşiği
_MAPE_ZERO_THRESH: float         = 1e-3  # |y_true| altı → MAPE dışı

# Health score ağırlıkları (toplam 1.0)
_HEALTH_WEIGHTS: Dict[str, float] = {
    "high_bias":         0.20,
    "heavy_tail":        0.10,
    "nonlinear":         0.10,
    "unstable_corr":     0.05,
    "poor_small_error":  0.20,
    "non_normal":        0.05,
    "low_r2":            0.15,
    "calibration":       0.10,
    "heteroscedastic":   0.05,
}


# ══════════════════════════════════════════════════════════════════════
# MetricResult
# ══════════════════════════════════════════════════════════════════════

@dataclass
class MetricResult:
    """
    Tek split için tüm metrikler, diagnostics ve health score.

    Alanlar gruplar halinde organize edilmiştir:
      1. Meta
      2. Temel hata
      3. Yüzde hata
      4. Hata dağılımı
      5. Bias
      6. Kalibrasyon
      7. Korelasyon
      8. Residual / dağılım
      9. Within-threshold
      10. Diagnostics — bias grubu
      11. Diagnostics — dağılım grubu
      12. Diagnostics — kalibrasyon grubu
      13. Composite health
    """

    # ── 1. Meta ─────────────────────────────────────────────────────
    prefix:    str = ""
    n_samples: int = 0

    # ── 2. Temel hata ────────────────────────────────────────────────
    mae:                float = float("nan")
    rmse:               float = float("nan")
    mse:                float = float("nan")
    medae:              float = float("nan")
    r2:                 float = float("nan")
    explained_variance: float = float("nan")

    # ── 3. Yüzde hata ───────────────────────────────────────────────
    mape:  float = float("nan")
    smape: float = float("nan")
    wape:  float = float("nan")

    # ── 4. Hata dağılımı ────────────────────────────────────────────
    mae_p50: float = float("nan")
    mae_p90: float = float("nan")
    mae_p95: float = float("nan")
    mae_p99: float = float("nan")
    max_err: float = float("nan")

    # ── 5. Bias ─────────────────────────────────────────────────────
    mean_bias:        float = float("nan")
    std_residuals:    float = float("nan")
    bias_tstat:       float = float("nan")
    bias_tpval:       float = float("nan")
    bias_ci_low:      float = float("nan")   # %95 CI alt
    bias_ci_high:     float = float("nan")   # %95 CI üst
    bias_effect_size: float = float("nan")   # |mean_bias| / std_res
    systematic_bias:  bool  = False          # p < α VE effect > eşik

    # ── 6. Kalibrasyon ──────────────────────────────────────────────
    regression_slope:     float = float("nan")   # ideal = 1
    regression_intercept: float = float("nan")   # ideal = 0
    overestimation_flag:  bool  = False   # slope < 1 - tol → yüksek değerleri küçümser
    underestimation_flag: bool  = False   # slope > 1 + tol → yüksek değerleri abartır

    # ── 7. Korelasyon ───────────────────────────────────────────────
    pearson_r:              float = float("nan")
    pearson_p:              float = float("nan")
    spearman_r:             float = float("nan")
    spearman_p:             float = float("nan")
    ccc:                    float = float("nan")
    pearson_spearman_gap:   float = float("nan")   # |pearson - spearman|

    # ── 8. Residual / dağılım ───────────────────────────────────────
    residual_skewness:      float = float("nan")
    residual_kurtosis:      float = float("nan")   # excess kurtosis
    normality_test:         str   = "none"
    normality_stat:         float = float("nan")
    normality_p:            float = float("nan")   # Anderson için NaN
    anderson_critical_5pct: float = float("nan")
    residuals_normal:       bool  = False
    heteroscedasticity_spearman_r: float = float("nan")
    heteroscedasticity_p:          float = float("nan")

    # ── 9. Within-threshold ─────────────────────────────────────────
    within_1min_pct:  float = float("nan")
    within_2min_pct:  float = float("nan")
    within_5min_pct:  float = float("nan")
    within_10min_pct: float = float("nan")

    # ── 10. Diagnostics — bias grubu ────────────────────────────────
    high_bias_flag: bool = False   # sistematik + pratik etki büyük

    # ── 11. Diagnostics — dağılım grubu ─────────────────────────────
    heavy_tail_flag:           bool = False   # kurtosis VE P99/P50 yüksek
    nonlinear_relationship_flag: bool = False # Spearman yüksek, Pearson düşük
    unstable_correlation_flag: bool = False   # Pearson/Spearman ayrışıyor
    poor_small_error_flag:     bool = False   # within_2min < %50
    non_normal_residuals_flag: bool = False
    heteroscedasticity_flag:   bool = False   # hata varyansı tahminle değişiyor

    # ── 12. Diagnostics — kalibrasyon grubu ─────────────────────────
    low_r2_flag:       bool = False
    calibration_flag:  bool = False   # slope ≠ 1 veya intercept büyük

    # ── 13. Composite health ────────────────────────────────────────
    health_score:  float = float("nan")   # 0 (kötü) – 1 (iyi)
    model_health:  str   = "unknown"      # "good" | "warning" | "critical"

    # ────────────────────────────────────────────────────────────────

    def to_dict(self, flat: bool = True) -> Dict:
        """
        flat=True  → prefix_key formatında düz dict (eski API uyumu).
        flat=False → ham dataclass dict.

        prefix = "" ise key'ler prefix'siz.
        """
        p   = f"{self.prefix}_" if self.prefix else ""
        raw = asdict(self)
        if not flat:
            return raw
        out: Dict = {}
        skip = {"prefix"}
        for k, v in raw.items():
            if k in skip:
                continue
            out[f"{p}{k}"] = v
        return out


# ══════════════════════════════════════════════════════════════════════
# Temizleme
# ══════════════════════════════════════════════════════════════════════

def _clean(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    NaN/inf temizle, uzunluk ve boyut kontrolü yap.
    n=0 sonrası → ValueError.
    n=1 → izin verilir; çağıran yönlendirir.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Şekil uyuşmazlığı: y_true={y_true.shape}, y_pred={y_pred.shape}"
        )
    if y_true.ndim != 1:
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()
    if len(y_true) == 0:
        raise ValueError("Boş dizi: en az 1 sample gerekli.")

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    n_rm = int((~mask).sum())
    if n_rm:
        warnings.warn(f"[evaluate] {n_rm} NaN/inf sample çıkarıldı.", stacklevel=3)

    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        raise ValueError("Temizleme sonrası hiç sample kalmadı.")
    return yt, yp


# ══════════════════════════════════════════════════════════════════════
# n = 1 özel durumu
# ══════════════════════════════════════════════════════════════════════

def _single_sample(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str,
) -> MetricResult:
    """
    n=1 için hesaplanabilecek alanları doldur.
    İstatistiksel metrikler NaN / False.
    """
    res     = float(y_true[0] - y_pred[0])
    abs_res = abs(res)
    denom   = abs(float(y_true[0])) + abs(float(y_pred[0])) + _EPS

    return MetricResult(
        prefix=prefix, n_samples=1,
        mae=abs_res, mse=abs_res**2, rmse=abs_res, max_err=abs_res,
        mean_bias=res,
        smape=float(200.0 * abs_res / denom),
        wape=float(abs_res / (abs(float(y_true[0])) + _EPS) * 100),
        within_1min_pct=100.0  if abs_res <= 1.0  else 0.0,
        within_2min_pct=100.0  if abs_res <= 2.0  else 0.0,
        within_5min_pct=100.0  if abs_res <= 5.0  else 0.0,
        within_10min_pct=100.0 if abs_res <= 10.0 else 0.0,
        model_health="unknown",
    )


# ══════════════════════════════════════════════════════════════════════
# Sayısal yardımcılar
# ══════════════════════════════════════════════════════════════════════

def _pearson(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < _MIN_CORR_N:
        return float("nan"), float("nan")
    sx, sy = float(np.std(x, ddof=1)), float(np.std(y, ddof=1))
    if sx < _EPS or sy < _EPS:
        return float("nan"), float("nan")
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def _spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < _MIN_CORR_N:
        return float("nan"), float("nan")
    if float(np.std(x, ddof=1)) < _EPS or float(np.std(y, ddof=1)) < _EPS:
        return float("nan"), float("nan")
    r, p = stats.spearmanr(x, y)
    return float(r), float(p)


def _ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Concordance Correlation Coefficient.
    ddof=0 (populasyon varyansı) — CCC tanımıyla tutarlı.
    Numerically stable: denom < _EPS → nan.
    """
    vt   = float(np.var(y_true, ddof=0))
    vp   = float(np.var(y_pred, ddof=0))
    mt   = float(np.mean(y_true))
    mp   = float(np.mean(y_pred))
    cov  = float(np.mean((y_true - mt) * (y_pred - mp)))
    denom = vt + vp + (mt - mp) ** 2
    if denom < _EPS:
        return float("nan")
    return float(2.0 * cov / denom)


def _mape(y_true: np.ndarray, residuals: np.ndarray) -> float:
    mask = np.abs(y_true) >= _MAPE_ZERO_THRESH
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(residuals[mask] / y_true[mask])) * 100)


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred) + _EPS
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100)


def _wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum(np.abs(y_true)))
    if denom < _EPS:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100)


def _bias_stats(
    residuals: np.ndarray,
) -> Tuple[float, float, float, float, float, float, bool]:
    """
    t-test (α=0.05) + effect size + %95 CI.
    systematic_bias = p < α VE effect_size > _BIAS_EFFECT_THRESH.

    Döner: tstat, tpval, ci_low, ci_high, effect_size, std_res, systematic_bias
    """
    n       = len(residuals)
    std_res = float(np.std(residuals, ddof=1))
    mean_b  = float(np.mean(residuals))

    if std_res < _EPS:
        # Sıfır varyans: tüm residuals aynı → t-test anlamsız
        return float("nan"), float("nan"), mean_b, mean_b, float("nan"), std_res, False

    try:
        tstat, tpval = stats.ttest_1samp(residuals, 0.0)
        tstat, tpval = float(tstat), float(tpval)
    except Exception:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), std_res, False

    se        = std_res / np.sqrt(n)
    t_crit    = float(stats.t.ppf(1.0 - _ALPHA / 2.0, df=n - 1))
    ci_low    = mean_b - t_crit * se
    ci_high   = mean_b + t_crit * se
    eff       = abs(mean_b) / (std_res + _EPS)
    sys_bias  = bool(tpval < _ALPHA and eff > _BIAS_EFFECT_THRESH)

    return tstat, tpval, float(ci_low), float(ci_high), float(eff), std_res, sys_bias


def _calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float, bool, bool, bool]:
    """
    OLS: y_pred ~ slope * y_true + intercept

    Döner: slope, intercept, calibration_flag,
           overestimation_flag, underestimation_flag

    overestimation : slope < 1 - tol → model yüksek değerleri küçümser
    underestimation: slope > 1 + tol → model yüksek değerleri abartır
    calibration_flag: slope veya intercept tolerans dışı
    """
    if float(np.std(y_true, ddof=1)) < _EPS:
        return float("nan"), float("nan"), False, False, False

    slope, intercept, *_ = stats.linregress(y_true, y_pred)
    slope     = float(slope)
    intercept = float(intercept)

    std_res = float(np.std(y_true - y_pred, ddof=1))
    over_est  = bool(slope < 1.0 - _CALIB_SLOPE_TOL)
    under_est = bool(slope > 1.0 + _CALIB_SLOPE_TOL)
    intercept_large = bool(abs(intercept) > _CALIB_INTERCEPT_RATIO * (std_res + _EPS))
    calib_flag = bool(over_est or under_est or intercept_large)

    return slope, intercept, calib_flag, over_est, under_est


def _normality(
    residuals: np.ndarray,
) -> Tuple[str, float, float, float, bool]:
    """
    n ≤ 5 000 → Shapiro-Wilk  (p gerçek)
    n ≤ 25 000 → Anderson-Darling (normality_p = NaN, kritik değer karşılaştırması)
    n > 25 000 → atla

    Döner: test_adı, stat, p_value, anderson_critical_5pct, residuals_normal
    """
    n = len(residuals)
    if n < 3:
        return "none", float("nan"), float("nan"), float("nan"), False

    if n <= _SHAPIRO_MAX_N:
        try:
            stat, p = stats.shapiro(residuals)
            return "shapiro", float(stat), float(p), float("nan"), bool(float(p) > _ALPHA)
        except Exception:
            pass

    if n <= _ANDERSON_MAX_N:
        try:
            res    = stats.anderson(residuals, dist="norm")
            stat   = float(res.statistic)
            crit5  = float(res.critical_values[2])   # %5 index
            return "anderson-darling", stat, float("nan"), crit5, bool(stat <= crit5)
        except Exception:
            pass

    return "skipped_large_n", float("nan"), float("nan"), float("nan"), False


def _heteroscedasticity(
    residuals: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float, bool]:
    """
    Basit proxy: |residuals| ile y_pred arasındaki Spearman korelasyonu.
    Anlamlı pozitif korelasyon → tahmin büyüdükçe hata da büyüyor.

    Döner: spearman_r, spearman_p, flag
    """
    if len(residuals) < _MIN_CORR_N:
        return float("nan"), float("nan"), False
    if float(np.std(np.abs(residuals), ddof=1)) < _EPS:
        return float("nan"), float("nan"), False

    r, p = stats.spearmanr(np.abs(residuals), y_pred)
    flag = bool(np.isfinite(p) and float(p) < _HETERO_SPEARMAN_P and float(r) > 0)
    return float(r), float(p), flag


def _health(flags: Dict[str, bool]) -> Tuple[float, str]:
    """
    Ağırlıklı flag toplamından composite health score üret.
    score = 1 - weighted_sum_of_active_flags  (0 kötü, 1 iyi)

    model_health:
      score ≥ 0.80 → "good"
      score ≥ 0.55 → "warning"
      score <  0.55 → "critical"
    """
    penalty = sum(
        _HEALTH_WEIGHTS.get(k, 0.0)
        for k, active in flags.items()
        if active
    )
    score = float(max(0.0, 1.0 - penalty))
    if score >= 0.80:
        label = "good"
    elif score >= 0.55:
        label = "warning"
    else:
        label = "critical"
    return score, label


# ══════════════════════════════════════════════════════════════════════
# Ana hesap fonksiyonları
# ══════════════════════════════════════════════════════════════════════

def regression_metrics_full(
    y_true: List[float] | np.ndarray,
    y_pred: List[float] | np.ndarray,
    prefix: str = "",
) -> MetricResult:
    """
    Tam metrik hesabı → MetricResult döner.

    n=1 → azaltılmış metrik seti (_single_sample).
    n≥2 → tüm metrikler, diagnostics, health score.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    yt, yp = _clean(yt, yp)
    n = len(yt)

    if n == 1:
        return _single_sample(yt, yp, prefix)

    # ── Temel hesaplar ───────────────────────────────────────────────
    res     = yt - yp                        # residuals (true − pred)
    abs_res = np.abs(res)                    # bir kez hesapla

    mae     = float(mean_absolute_error(yt, yp))
    mse     = float(mean_squared_error(yt, yp))
    rmse    = float(np.sqrt(mse))
    medae   = float(median_absolute_error(yt, yp))
    r2      = float(r2_score(yt, yp))
    ev      = float(explained_variance_score(yt, yp))

    mape_v  = _mape(yt, res)
    smape_v = _smape(yt, yp)
    wape_v  = _wape(yt, yp)

    # ── Percentile'lar — tek çağrı ───────────────────────────────────
    p50, p90, p95, p99 = (
        float(v) for v in np.percentile(abs_res, [50, 90, 95, 99])
    )
    max_err = float(np.max(abs_res))

    # ── Bias ────────────────────────────────────────────────────────
    mean_b = float(np.mean(res))
    tstat, tpval, ci_lo, ci_hi, eff, std_res, sys_bias = _bias_stats(res)

    # ── Kalibrasyon ──────────────────────────────────────────────────
    slope, intercept, calib_flag, over_est, under_est = _calibration(yt, yp)

    # ── Korelasyon ───────────────────────────────────────────────────
    pr, pp   = _pearson(yt, yp)
    sr, sp   = _spearman(yt, yp)
    ccc_v    = _ccc(yt, yp)
    ps_gap   = abs(pr - sr) if (np.isfinite(pr) and np.isfinite(sr)) else float("nan")

    # ── Residual / dağılım ───────────────────────────────────────────
    skew_v   = float(stats.skew(res))
    kurt_v   = float(stats.kurtosis(res))   # excess kurtosis
    nt, ns, np_, and_crit, res_norm = _normality(res)
    hetero_r, hetero_p, hetero_flag = _heteroscedasticity(res, yp)

    # ── Within-threshold ────────────────────────────────────────────
    w1  = float(np.mean(abs_res <= 1.0)  * 100)
    w2  = float(np.mean(abs_res <= 2.0)  * 100)
    w5  = float(np.mean(abs_res <= 5.0)  * 100)
    w10 = float(np.mean(abs_res <= 10.0) * 100)

    # ── Diagnostics flags ───────────────────────────────────────────
    heavy_tail = bool(
        kurt_v > _HEAVY_TAIL_KURTOSIS
        and np.isfinite(p50) and p50 > _EPS
        and (p99 / (p50 + _EPS)) > _HEAVY_TAIL_P99_P50_RATIO
    )
    nonlinear = bool(
        np.isfinite(sr) and abs(sr) >= _NONLINEAR_SPEARMAN_MIN
        and np.isfinite(pr)
        and (abs(sr) - abs(pr)) >= _NONLINEAR_PEARSON_GAP
    )
    unstable_corr = bool(
        np.isfinite(ps_gap)
        and ps_gap > _UNSTABLE_CORR_GAP
        and np.isfinite(pr)
        and abs(pr) < _UNSTABLE_CORR_ABS
    )
    non_normal = bool(
        not res_norm and nt not in ("skipped_large_n", "none")
    )
    low_r2 = bool(r2 < 0.70)

    flag_map = {
        "high_bias":        sys_bias,
        "heavy_tail":       heavy_tail,
        "nonlinear":        nonlinear,
        "unstable_corr":    unstable_corr,
        "poor_small_error": w2 < 50.0,
        "non_normal":       non_normal,
        "low_r2":           low_r2,
        "calibration":      calib_flag,
        "heteroscedastic":  hetero_flag,
    }
    h_score, h_label = _health(flag_map)

    return MetricResult(
        prefix=prefix, n_samples=n,

        mae=mae, rmse=rmse, mse=mse, medae=medae,
        r2=r2, explained_variance=ev,

        mape=mape_v, smape=smape_v, wape=wape_v,

        mae_p50=p50, mae_p90=p90, mae_p95=p95, mae_p99=p99,
        max_err=max_err,

        mean_bias=mean_b, std_residuals=std_res,
        bias_tstat=tstat, bias_tpval=tpval,
        bias_ci_low=ci_lo, bias_ci_high=ci_hi,
        bias_effect_size=eff, systematic_bias=sys_bias,

        regression_slope=slope, regression_intercept=intercept,
        overestimation_flag=over_est, underestimation_flag=under_est,

        pearson_r=pr, pearson_p=pp,
        spearman_r=sr, spearman_p=sp,
        ccc=ccc_v, pearson_spearman_gap=ps_gap,

        residual_skewness=skew_v, residual_kurtosis=kurt_v,
        normality_test=nt, normality_stat=ns, normality_p=np_,
        anderson_critical_5pct=and_crit, residuals_normal=res_norm,
        heteroscedasticity_spearman_r=hetero_r,
        heteroscedasticity_p=hetero_p,

        within_1min_pct=w1, within_2min_pct=w2,
        within_5min_pct=w5, within_10min_pct=w10,

        # Bias grubu
        high_bias_flag=sys_bias,
        # Dağılım grubu
        heavy_tail_flag=heavy_tail,
        nonlinear_relationship_flag=nonlinear,
        unstable_correlation_flag=unstable_corr,
        poor_small_error_flag=bool(w2 < 50.0),
        non_normal_residuals_flag=non_normal,
        heteroscedasticity_flag=hetero_flag,
        # Kalibrasyon grubu
        low_r2_flag=low_r2,
        calibration_flag=calib_flag,

        health_score=h_score,
        model_health=h_label,
    )


def regression_metrics(
    y_true: List[float] | np.ndarray,
    y_pred: List[float] | np.ndarray,
    prefix: str = "",
) -> Dict:
    """
    Eski API uyumu: düz dict döner.
    Zengin erişim için regression_metrics_full() kullan.
    """
    return regression_metrics_full(y_true, y_pred, prefix=prefix).to_dict(flat=True)


# ══════════════════════════════════════════════════════════════════════
# Split karşılaştırma
# ══════════════════════════════════════════════════════════════════════

@dataclass
class SplitComparison:
    """
    Train / val / test karşılaştırma.
    overfit_level rule-based heuristic'tir — istatistiksel test değildir.
    """

    train: MetricResult
    val:   MetricResult
    test:  MetricResult

    mae_gap_train_val:  float = float("nan")
    mae_gap_val_test:   float = float("nan")
    r2_drop_train_test: float = float("nan")
    rmse_gap_train_val: float = float("nan")

    overfit_level: str = "none"   # "none" | "mild" | "moderate" | "severe"
    overfit_note:  str = (
        "Heuristic rule-based assessment — not a formal statistical test."
    )

    any_bias:           bool = False
    any_heavy_tail:     bool = False
    any_low_r2:         bool = False
    any_poor_accuracy:  bool = False
    any_heteroscedastic: bool = False
    generalization_ok:  bool = True

    def to_dict(self) -> Dict:
        return {
            "train":   self.train.to_dict(flat=True),
            "val":     self.val.to_dict(flat=True),
            "test":    self.test.to_dict(flat=True),
            "summary": {
                "mae_gap_train_val":   self.mae_gap_train_val,
                "mae_gap_val_test":    self.mae_gap_val_test,
                "r2_drop_train_test":  self.r2_drop_train_test,
                "rmse_gap_train_val":  self.rmse_gap_train_val,
                "overfit_level":       self.overfit_level,
                "overfit_note":        self.overfit_note,
                "any_bias":            self.any_bias,
                "any_heavy_tail":      self.any_heavy_tail,
                "any_low_r2":          self.any_low_r2,
                "any_poor_accuracy":   self.any_poor_accuracy,
                "any_heteroscedastic": self.any_heteroscedastic,
                "generalization_ok":   self.generalization_ok,
            },
        }


def compare_splits(
    train_true, train_pred,
    val_true,   val_pred,
    test_true,  test_pred,
) -> SplitComparison:
    """
    Üç split için MetricResult hesapla, karşılaştır.

    Overfit seviyeleri (heuristic eşikler):
      severe   : ΔR² > 0.15 VEYA ΔMAE > 2.0 dk
      moderate : ΔR² > 0.08 VEYA ΔMAE > 1.0 dk
      mild     : ΔR² > 0.03 VEYA ΔMAE > 0.4 dk
      none     : aksi
    """
    tm = regression_metrics_full(train_true, train_pred, prefix="train")
    vm = regression_metrics_full(val_true,   val_pred,   prefix="val")
    em = regression_metrics_full(test_true,  test_pred,  prefix="test")

    gap_tv   = tm.mae  - vm.mae
    gap_vt   = vm.mae  - em.mae
    r2_drop  = tm.r2   - em.r2
    rmse_gap = tm.rmse - vm.rmse
    abs_gap  = abs(gap_tv)

    if   r2_drop > 0.15 or abs_gap > 2.0: level = "severe"
    elif r2_drop > 0.08 or abs_gap > 1.0: level = "moderate"
    elif r2_drop > 0.03 or abs_gap > 0.4: level = "mild"
    else:                                  level = "none"

    vm_mae = vm.mae
    gen_ok = bool(vm_mae < _EPS or abs(vm_mae - em.mae) / (vm_mae + _EPS) < 0.20)

    return SplitComparison(
        train=tm, val=vm, test=em,
        mae_gap_train_val=float(gap_tv),
        mae_gap_val_test=float(gap_vt),
        r2_drop_train_test=float(r2_drop),
        rmse_gap_train_val=float(rmse_gap),
        overfit_level=level,
        any_bias=any([tm.high_bias_flag,         vm.high_bias_flag,         em.high_bias_flag]),
        any_heavy_tail=any([tm.heavy_tail_flag,   vm.heavy_tail_flag,        em.heavy_tail_flag]),
        any_low_r2=any([tm.low_r2_flag,           vm.low_r2_flag,            em.low_r2_flag]),
        any_poor_accuracy=any([tm.poor_small_error_flag, vm.poor_small_error_flag, em.poor_small_error_flag]),
        any_heteroscedastic=any([tm.heteroscedasticity_flag, vm.heteroscedasticity_flag, em.heteroscedasticity_flag]),
        generalization_ok=gen_ok,
    )


# ══════════════════════════════════════════════════════════════════════
# Print
# ══════════════════════════════════════════════════════════════════════

def _row(label: str, val: str, warn: bool = False) -> None:
    prefix = "  ⚠ " if warn else "    "
    print(f"{prefix}{label:<32}: {val}")


def _sec(title: str) -> None:
    print(f"\n  ── {title}")


def _fmt(v: float, decimals: int = 4, unit: str = "") -> str:
    return f"{v:.{decimals}f}{(' ' + unit) if unit else ''}" if np.isfinite(v) else "n/a"


def print_metrics(
    metrics: MetricResult | Dict,
    title: str = "Metrics",
) -> None:
    """
    MetricResult veya düz dict (geriye dönük uyum) kabul eder.
    Kritik uyarılar ⚠ ile öne çıkarılır; bağlam yorumu üretilir.
    """
    if isinstance(metrics, dict):
        sample_key = next(iter(metrics), "")
        pfx = ""
        for s in ("train_", "val_", "test_"):
            if sample_key.startswith(s):
                pfx = s.rstrip("_")
                break
        m = _dict_to_mr(metrics, pfx)
    else:
        m = metrics

    W = 65
    print(f"\n{'═' * W}")
    print(f"  {title}  (n={m.n_samples})  "
          f"health={m.model_health.upper()}  score={_fmt(m.health_score, 2)}")
    if m.n_samples == 1:
        print("  ⚠  n=1 — istatistiksel metrikler NaN.")
    print(f"{'═' * W}")

    _sec("Temel hata")
    _row("MAE",           _fmt(m.mae,  4, "dk"))
    _row("RMSE",          _fmt(m.rmse, 4, "dk"))
    _row("Median AE",     _fmt(m.medae, 4, "dk"))
    _row("R²",            _fmt(m.r2),  warn=m.low_r2_flag)
    _row("Explained Var", _fmt(m.explained_variance))
    _row("sMAPE",         _fmt(m.smape, 2, "%"))
    _row("WAPE",          _fmt(m.wape,  2, "%"))
    _row("MAPE",          _fmt(m.mape,  2, "%"))

    _sec("Hata dağılımı (abs residuals)")
    _row("P50",  _fmt(m.mae_p50, 4, "dk"))
    _row("P90",  _fmt(m.mae_p90, 4, "dk"))
    _row("P95",  _fmt(m.mae_p95, 4, "dk"))
    _row("P99",  _fmt(m.mae_p99, 4, "dk"), warn=m.heavy_tail_flag)
    _row("Max",  _fmt(m.max_err, 4, "dk"))

    _sec("±N dakika içinde (%)")
    _row("±1 dk",  _fmt(m.within_1min_pct,  1, "%"))
    _row("±2 dk",  _fmt(m.within_2min_pct,  1, "%"), warn=m.poor_small_error_flag)
    _row("±5 dk",  _fmt(m.within_5min_pct,  1, "%"))
    _row("±10 dk", _fmt(m.within_10min_pct, 1, "%"))

    _sec("Korelasyon")
    corr_warn = m.unstable_correlation_flag or m.nonlinear_relationship_flag
    _row("Pearson r",            _fmt(m.pearson_r)  + f"  p={_fmt(m.pearson_p, 2)}")
    _row("Spearman r",           _fmt(m.spearman_r) + f"  p={_fmt(m.spearman_p, 2)}", warn=corr_warn)
    _row("CCC",                  _fmt(m.ccc))
    _row("|Pearson−Spearman|",   _fmt(m.pearson_spearman_gap))
    if m.nonlinear_relationship_flag:
        print("       → Spearman >> Pearson: muhtemel nonlinear ilişki")
    if m.unstable_correlation_flag:
        print("       → Korelasyon kararsız: outlier veya dağılım sorunu olabilir")

    bias_ci = (
        f"[%95 CI: {_fmt(m.bias_ci_low, 4)}, {_fmt(m.bias_ci_high, 4)}]  "
        f"effect={_fmt(m.bias_effect_size, 3)}"
        if np.isfinite(m.bias_ci_low) else ""
    )
    _sec("Bias analizi")
    _row("Mean bias",        _fmt(m.mean_bias, 4, "dk") + f"  {bias_ci}", warn=m.high_bias_flag)
    _row("Std residuals",    _fmt(m.std_residuals, 4, "dk"))
    _row("Sistematik bias",  "EVET ⚠" if m.systematic_bias else "Hayır")

    _sec("Kalibrasyon")
    calib_detail = f"slope={_fmt(m.regression_slope, 4)}  intercept={_fmt(m.regression_intercept, 4)}"
    _row("OLS fit (pred~true)", calib_detail, warn=m.calibration_flag)
    if m.overestimation_flag:
        print("       → slope < 1: model yüksek gecikmeleri KÜÇÜMSÜYOR (overestimation pattern)")
    if m.underestimation_flag:
        print("       → slope > 1: model yüksek gecikmeleri ABARTIYOR (underestimation pattern)")
    _row("Heteroscedasticity",
         f"Spearman r={_fmt(m.heteroscedasticity_spearman_r, 3)}  "
         f"p={_fmt(m.heteroscedasticity_p, 3)}",
         warn=m.heteroscedasticity_flag)
    if m.heteroscedasticity_flag:
        print("       → Hata varyansı büyük tahminlerde artıyor")

    if m.normality_test == "anderson-darling":
        norm_str = (
            f"anderson-darling  stat={_fmt(m.normality_stat, 4)}  "
            f"crit_5%={_fmt(m.anderson_critical_5pct, 4)}  p=N/A"
        )
    elif np.isfinite(m.normality_stat):
        norm_str = (
            f"{m.normality_test}  stat={_fmt(m.normality_stat, 4)}  "
            f"p={_fmt(m.normality_p, 4)}"
        )
    else:
        norm_str = m.normality_test

    _sec("Residual istatistikleri")
    _row("Skewness",       _fmt(m.residual_skewness, 4))
    _row("Kurtosis (exc)", _fmt(m.residual_kurtosis, 4), warn=m.heavy_tail_flag)
    _row("Normallik",      norm_str, warn=m.non_normal_residuals_flag)
    _row("Normal mi?",     "Evet" if m.residuals_normal else "Hayır")

    # Gruplu diagnostics özeti
    bias_flags   = {"high_bias": m.high_bias_flag}
    dist_flags   = {
        "heavy_tail":    m.heavy_tail_flag,
        "nonlinear":     m.nonlinear_relationship_flag,
        "unstable_corr": m.unstable_correlation_flag,
        "poor_accuracy": m.poor_small_error_flag,
        "non_normal":    m.non_normal_residuals_flag,
        "hetero":        m.heteroscedasticity_flag,
    }
    calib_flags  = {"low_r2": m.low_r2_flag, "calibration": m.calibration_flag}

    _sec("Diagnostics özeti (heuristic)")
    for grp_name, grp in [("Bias", bias_flags), ("Dağılım", dist_flags), ("Kalibrasyon", calib_flags)]:
        active = [k for k, v in grp.items() if v]
        status = ("⚠  " + ", ".join(active)) if active else "✓  temiz"
        print(f"    {grp_name:<14}: {status}")

    print(f"\n  Model health : {m.model_health.upper()}  (score={_fmt(m.health_score, 3)})")
    print(f"{'═' * W}\n")


def print_comparison(cmp: SplitComparison) -> None:
    W = 74
    print(f"\n{'═' * W}")
    print(f"  SPLIT KARŞILAŞTIRMA")
    print(f"{'═' * W}")
    h = f"  {'Metrik':<26} {'Train':>10} {'Val':>10} {'Test':>10}"
    print(h)
    print(f"  {'-' * 56}")

    rows: List[Tuple] = [
        ("MAE (dk)",        cmp.train.mae,           cmp.val.mae,           cmp.test.mae),
        ("RMSE (dk)",       cmp.train.rmse,           cmp.val.rmse,          cmp.test.rmse),
        ("R²",              cmp.train.r2,             cmp.val.r2,            cmp.test.r2),
        ("CCC",             cmp.train.ccc,            cmp.val.ccc,           cmp.test.ccc),
        ("Spearman r",      cmp.train.spearman_r,     cmp.val.spearman_r,    cmp.test.spearman_r),
        ("within_2min (%)", cmp.train.within_2min_pct, cmp.val.within_2min_pct, cmp.test.within_2min_pct),
        ("within_5min (%)", cmp.train.within_5min_pct, cmp.val.within_5min_pct, cmp.test.within_5min_pct),
        ("Mean bias (dk)",  cmp.train.mean_bias,      cmp.val.mean_bias,     cmp.test.mean_bias),
        ("Health score",    cmp.train.health_score,   cmp.val.health_score,  cmp.test.health_score),
    ]
    for name, tr, va, te in rows:
        print(f"  {name:<26} {_fmt(tr):>10} {_fmt(va):>10} {_fmt(te):>10}")

    health_row = f"  {'Model health':<26} {cmp.train.model_health:>10} {cmp.val.model_health:>10} {cmp.test.model_health:>10}"
    print(health_row)

    print(f"\n  {'Gap (train−val MAE)':<34}: {cmp.mae_gap_train_val:+.4f} dk")
    print(f"  {'Gap (val−test MAE)':<34}: {cmp.mae_gap_val_test:+.4f} dk")
    print(f"  {'R² düşüşü (train→test)':<34}: {cmp.r2_drop_train_test:+.4f}")
    print(f"  {'Overfit (heuristic)':<34}: {cmp.overfit_level.upper()}")
    print(f"  {'  ↳':<34}: {cmp.overfit_note}")
    print(f"  {'Generalization OK':<34}: {'Evet' if cmp.generalization_ok else 'Hayır ⚠'}")

    warns = [
        (cmp.any_bias,            "En az bir split'te sistematik bias"),
        (cmp.any_heavy_tail,      "En az bir split'te ağır kuyruk"),
        (cmp.any_low_r2,          "En az bir split'te R² < 0.70"),
        (cmp.any_poor_accuracy,   "En az bir split'te within_2min < %50"),
        (cmp.any_heteroscedastic, "En az bir split'te heteroscedasticity"),
    ]
    active_w = [msg for flag, msg in warns if flag]
    if active_w:
        print()
        for w in active_w:
            print(f"  ⚠  {w}")

    print(f"{'═' * W}\n")


# ══════════════════════════════════════════════════════════════════════
# Geriye dönük uyum yardımcısı
# ══════════════════════════════════════════════════════════════════════

def _dict_to_mr(d: Dict, prefix: str) -> MetricResult:
    """Düz dict → MetricResult (yalnızca print_metrics geriye dönük uyumu için)."""
    p = f"{prefix}_" if prefix else ""

    def g(k: str, default=float("nan")):
        return d.get(f"{p}{k}", default)

    return MetricResult(
        prefix=prefix, n_samples=int(g("n_samples", 0)),
        mae=g("mae"), rmse=g("rmse"), mse=g("mse"), medae=g("medae"),
        r2=g("r2"), explained_variance=g("explained_variance"),
        mape=g("mape"), smape=g("smape"), wape=g("wape"),
        mae_p50=g("mae_p50"), mae_p90=g("mae_p90"),
        mae_p95=g("mae_p95"), mae_p99=g("mae_p99"), max_err=g("max_err"),
        mean_bias=g("mean_bias"), std_residuals=g("std_residuals"),
        bias_tstat=g("bias_tstat"), bias_tpval=g("bias_tpval"),
        bias_ci_low=g("bias_ci_low"), bias_ci_high=g("bias_ci_high"),
        bias_effect_size=g("bias_effect_size"),
        systematic_bias=bool(g("systematic_bias", False)),
        regression_slope=g("regression_slope"),
        regression_intercept=g("regression_intercept"),
        overestimation_flag=bool(g("overestimation_flag", False)),
        underestimation_flag=bool(g("underestimation_flag", False)),
        pearson_r=g("pearson_r"), pearson_p=g("pearson_p"),
        spearman_r=g("spearman_r"), spearman_p=g("spearman_p"),
        ccc=g("ccc"), pearson_spearman_gap=g("pearson_spearman_gap"),
        residual_skewness=g("residual_skewness"),
        residual_kurtosis=g("residual_kurtosis"),
        normality_test=str(g("normality_test", "none")),
        normality_stat=g("normality_stat"),
        normality_p=g("normality_p"),
        anderson_critical_5pct=g("anderson_critical_5pct"),
        residuals_normal=bool(g("residuals_normal", False)),
        heteroscedasticity_spearman_r=g("heteroscedasticity_spearman_r"),
        heteroscedasticity_p=g("heteroscedasticity_p"),
        within_1min_pct=g("within_1min_pct"), within_2min_pct=g("within_2min_pct"),
        within_5min_pct=g("within_5min_pct"), within_10min_pct=g("within_10min_pct"),
        high_bias_flag=bool(g("high_bias_flag", False)),
        heavy_tail_flag=bool(g("heavy_tail_flag", False)),
        nonlinear_relationship_flag=bool(g("nonlinear_relationship_flag", False)),
        unstable_correlation_flag=bool(g("unstable_correlation_flag", False)),
        poor_small_error_flag=bool(g("poor_small_error_flag", False)),
        non_normal_residuals_flag=bool(g("non_normal_residuals_flag", False)),
        heteroscedasticity_flag=bool(g("heteroscedasticity_flag", False)),
        low_r2_flag=bool(g("low_r2_flag", False)),
        calibration_flag=bool(g("calibration_flag", False)),
        health_score=g("health_score"),
        model_health=str(g("model_health", "unknown")),
    )