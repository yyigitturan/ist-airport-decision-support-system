from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd


@dataclass
class SplitConfig:
    input_path: str = "/Users/YGT/ist-airport-decision-support-system/data/final/full_df.parquet"
    output_dir: str = "/Users/YGT/ist-airport-decision-support-system/data/model/splits"

    target_col: str = "post_terminal_duration_min"
    time_col: str = "obs_time"
    group_col: str = "focusing_flight_id"

    train_ratio: float = 0.80
    valid_ratio: float = 0.10
    test_ratio: float = 0.10

    min_target: float = 0.0
    max_target: float = 180.0

    drop_na_target: bool = True
    drop_na_time: bool = True
    drop_na_group: bool = True

    min_flights_per_month: int = 1000
    save_clean_full: bool = True




def basic_cleaning(df: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    df = df.copy()
    df[cfg.time_col] = pd.to_datetime(df[cfg.time_col], utc=True, errors="coerce")
    if "entry_time" in df.columns:
        df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
    if "arr_sched_time_utc" in df.columns:
        df["arr_sched_time_utc"] = pd.to_datetime(df["arr_sched_time_utc"], utc=True, errors="coerce")

    for col, flag in [(cfg.time_col, cfg.drop_na_time), (cfg.target_col, cfg.drop_na_target), (cfg.group_col, cfg.drop_na_group)]:
        if flag:
            before = len(df)
            df = df.dropna(subset=[col])
            print(f"  drop_na {col}: {before - len(df):,} satır silindi")

    before = len(df)
    df = df[(df[cfg.target_col] >= cfg.min_target) & (df[cfg.target_col] <= cfg.max_target)].copy()
    print(f"  target range filter: {before - len(df):,} has been removed")

    return df.sort_values(cfg.time_col).reset_index(drop=True)


def split_by_group_time_order(
    df: pd.DataFrame, group_col: str,
    train_ratio: float, valid_ratio: float, test_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unique_groups = (
        df[[group_col, "obs_time"]]
        .groupby(group_col, as_index=False)["obs_time"].min()
        .sort_values("obs_time").reset_index(drop=True)
    )
    n = len(unique_groups)
    t_end = int(n * train_ratio)
    v_end = int(n * (train_ratio + valid_ratio))

    train_g = set(unique_groups.iloc[:t_end][group_col])
    valid_g = set(unique_groups.iloc[t_end:v_end][group_col])
    test_g  = set(unique_groups.iloc[v_end:][group_col])

    return (
        df[df[group_col].isin(train_g)].copy(),
        df[df[group_col].isin(valid_g)].copy(),
        df[df[group_col].isin(test_g)].copy(),
    )


def leakage_check(train_df, valid_df, test_df, group_col):
    ti, vi, si = set(train_df[group_col]), set(valid_df[group_col]), set(test_df[group_col])
    tv, tt, vt = ti & vi, ti & si, vi & si
    if tv or tt or vt:
        raise RuntimeError(f"Leakage! train∩valid={len(tv)} train∩test={len(tt)} valid∩test={len(vt)}")
    print(" Leakage check passed")


def print_summary(name: str, df: pd.DataFrame, cfg: SplitConfig) -> None:
    print(f"  {name:6s} | rows={len(df):>7,} | flights={df[cfg.group_col].nunique():>5,} | "
          f"label mean={df[cfg.target_col].mean():.1f} std={df[cfg.target_col].std():.1f}")


def save_split(label: str, df: pd.DataFrame, out_dir: Path, cfg: SplitConfig) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df, valid_df, test_df = split_by_group_time_order(
        df, cfg.group_col, cfg.train_ratio, cfg.valid_ratio, cfg.test_ratio
    )
    leakage_check(train_df, valid_df, test_df, cfg.group_col)
    print_summary("TRAIN", train_df, cfg)
    print_summary("VALID", valid_df, cfg)
    print_summary("TEST",  test_df,  cfg)
    train_df.to_parquet(out_dir / "train.parquet", index=False)
    valid_df.to_parquet(out_dir / "valid.parquet", index=False)
    test_df.to_parquet( out_dir / "test.parquet",  index=False)
    print(f" [{label}] → {out_dir}")


def main() -> None:
    cfg = SplitConfig()

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(cfg.input_path)
    print(f"shape: {df.shape}")
    df = basic_cleaning(df, cfg)
    print(f"After Cleaning Shape: {df.shape}")

    if cfg.save_clean_full:
        p = output_dir / "full_df_clean.parquet"
        df.to_parquet(p, index=False)


    print(f"\n{'='*55}")
    print(f"  ALL | {len(df):,} scenarios | {df[cfg.group_col].nunique():,} flights")
    print(f"{'='*55}")
    save_split("all", df, output_dir / "all", cfg)

    # ── Monthly → splits/monthly/YYYY-MM/
    df["_month"] = df[cfg.time_col].dt.to_period("M").astype(str)
    monthly_flights   = df.groupby("_month")[cfg.group_col].nunique().rename("n_flights")
    monthly_scenarios = df.groupby("_month").size().rename("n_scenarios")
    monthly_summary   = pd.concat([monthly_flights, monthly_scenarios], axis=1).sort_index()

    print(f"\n{'='*55}")
    print("MONTHLY SUMMARY")
    print(f"{'Month':<12} {'Flights':>8} {'Scenarios':>10} {'Process':>8}")
    print("─" * 42)

    qualifying = []
    for month_str, r in monthly_summary.iterrows():
        n_f, n_s = int(r["n_flights"]), int(r["n_scenarios"])
        ok = n_f >= cfg.min_flights_per_month
        print(f"{month_str:<12} {n_f:>8,} {n_s:>10,} {'ok' if ok else 'skip':>8}")
        if ok:
            qualifying.append(month_str)

    print(f"\nQualifying months ({len(qualifying)}): {qualifying}")

    for month_str in qualifying:
        month_df = df[df["_month"] == month_str].drop(columns=["_month"]).copy()
        print(f"\n{'='*55}")
        print(f"  {month_str} | {len(month_df):,} scenarios | {month_df[cfg.group_col].nunique():,} flights")
        print(f"{'='*55}")
        save_split(month_str, month_df, output_dir / "monthly" / month_str, cfg)

    print(f"\n{'='*55}")
    print(f"  splits/monthly/ → {len(qualifying)} ay")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()