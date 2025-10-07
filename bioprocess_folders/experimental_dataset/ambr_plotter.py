
"""
ambr_plotter.py — Utilities to load AMBR metadata + time-series CSVs and plot time series.

Usage example:
    from ambr_plotter import load_metadata, load_runs, link_metadata, plot_timeseries

    meta = load_metadata("ambr.xlsx")
    runs = load_runs(["ambr_run1_140323_13-18.csv", "ambr_run1_140323_19-24.csv"])
    df = link_metadata(runs, meta)
    # Plot DO and OD for reactors 13-18
    plot_timeseries(df, variables=["DO", "Optical density"], bioreactors=[13,14,15,16,17,18])
"""

from __future__ import annotations
import re
from typing import List, Optional, Dict, Tuple
import pandas as pd
import matplotlib.pyplot as plt

# --- Units dictionary (extend as needed)
UNITS: Dict[str, str] = {
    "Time": "h",
    "Acid volume pumped": "mL",
    "Air flow": "mL/min",          # instrument may store sccm; edit if needed
    "Base volume pumped": "mL",
    "Bioreactor pressure reading": "mbar",  # adjust if psi
    "CER": "mmol/h",
    "DO": "%",
    "Feed#1 flow rate": "mL/h",
    "Feed#1 volume pumped": "mL",
    "Feed#2 flow rate": "mL/h",
    "Feed#2 volume pumped": "mL",
    "Foam sensor": "-",
    "Off-gas CO2%": "%",
    "Optical density": "-",
    "OUR": "mmol/h",
    "pH": "-",
    "Reflectance": "a.u.",
    "RQ": "-",
    "Sampling events": "-",
    "Stir speed": "rpm",
    "Temperature": "°C",
    "Volume": "mL",
}

def _parse_bioreactor_column(col: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Parse columns like 'Bioreactor 13 - DO' -> (13, 'DO').
    Returns (None, None) if not a bioreactor column.
    """
    m = re.match(r"^Bioreactor\s+(\d+)\s*-\s*(.+)$", col)
    if m:
        return int(m.group(1)), m.group(2).strip()
    return None, None

def load_metadata(path: str) -> pd.DataFrame:
    """
    Load metadata Excel file. Returns a dataframe with at least columns:
    ['Tunniste', 'StartDate', ...].
    """
    xl = pd.ExcelFile(path)
    # pick first sheet by default
    df = xl.parse(xl.sheet_names[0])
    # Normalize a key to join on (AMBR_13 -> 13)
    def tunniste_to_id(x):
        m = re.search(r"(\d+)$", str(x))
        return int(m.group(1)) if m else None
    df["BioreactorID"] = df["Tunniste"].apply(tunniste_to_id)
    return df

def load_runs(paths: List[str]) -> pd.DataFrame:
    """
    Load one or more AMBR CSV files into a tidy long dataframe with columns:
    ['Run', 'Time', 'BioreactorID', 'Variable', 'Value']
    Run will be derived from filename stem.
    """
    frames = []
    for p in paths:
        run_name = _filename_stem(p)
        raw = pd.read_csv(p)
        if "Time" not in raw.columns:
            raise ValueError(f"'Time' column not found in {p}")
        time_col = raw["Time"].astype(float)
        # Collect bioreactor columns and melt
        data_cols = [c for c in raw.columns if c != "Time"]
        records = []
        for c in data_cols:
            rid, var = _parse_bioreactor_column(c)
            if rid is not None and var is not None:
                records.append((rid, var, raw[c].values))
        if not records:
            continue
        # Build long dataframe
        rows = []
        for rid, var, vals in records:
            rows.append(pd.DataFrame({
                "Run": run_name,
                "Time": time_col,
                "BioreactorID": rid,
                "Variable": var,
                "Value": vals
            }))
        df_run = pd.concat(rows, ignore_index=True)
        frames.append(df_run)
    if not frames:
        raise ValueError("No bioreactor columns found in provided CSV files.")
    df = pd.concat(frames, ignore_index=True)
    # Attach unit for convenience
    df["Unit"] = df["Variable"].map(UNITS).fillna("")
    return df

def _filename_stem(path: str) -> str:
    import os
    base = os.path.basename(path)
    return os.path.splitext(base)[0]

def link_metadata(long_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join long_df with metadata on BioreactorID to add context fields such as
    StartDate, InitialOD, Medium, etc. Returns the augmented dataframe.
    """
    # Choose a subset of useful metadata columns if present
    keep_cols = [c for c in [
        "BioreactorID", "Tunniste", "StartDate", "Organism", "Strain", "Preculture",
        "InitialOD", "VesselType", "VesselVolume", "LiquidVolume",
        "Temperature", "ShakerRPM", "CarbonSource", "InitialConcentration(g/L)",
        "Medium", "InitialPH"
    ] if c in meta_df.columns or c == "BioreactorID"]
    meta_sub = meta_df[keep_cols].drop_duplicates(subset=["BioreactorID"])
    merged = long_df.merge(meta_sub, on="BioreactorID", how="left")
    return merged

def available_variables(long_df: pd.DataFrame) -> list:
    """Return a sorted list of available variables in the long dataframe."""
    return sorted(long_df["Variable"].unique())

def available_bioreactors(long_df: pd.DataFrame) -> list:
    """Return a sorted list of available bioreactor IDs in the long dataframe."""
    return sorted(long_df["BioreactorID"].unique())

def plot_timeseries(long_df,
                    variables=None,
                    bioreactors=None,
                    run=None,
                    sharex=True):
    import numpy as np
    import matplotlib.pyplot as plt

    df = long_df.copy()

    # Ensure clean dtypes
    if "Time" in df.columns:
        df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    if "Value" in df.columns:
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    # Text columns as strings (avoid float/NaN creeping in)
    for col in ("Run", "Variable", "Unit"):
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Optional filters
    if run is not None:
        df = df[df["Run"] == str(run)]

    if variables is None:
        variables = sorted(df["Variable"].unique())
    else:
        variables = [str(v) for v in variables]

    if bioreactors is None:
        bioreactors = sorted(df["BioreactorID"].dropna().astype(int).unique())
    else:
        bioreactors = [int(b) for b in bioreactors]

    for var in variables:
        sub = df[(df["Variable"] == var) & (df["BioreactorID"].astype(float).isin(bioreactors))]
        # Keep only numeric rows
        sub = sub.dropna(subset=["Time", "Value"])
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(9, 4.5))

        for rid in sorted(sub["BioreactorID"].dropna().astype(int).unique()):
            sub_r = sub[sub["BioreactorID"].astype(int) == rid].sort_values("Time")
            # Final safety filter per series
            x = sub_r["Time"].to_numpy(dtype=float)
            y = sub_r["Value"].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.any():
                ax.plot(x[mask], y[mask], label=f"BR {rid}")

        # Unit label
        unit_vals = [u for u in sub["Unit"].unique() if u and u != "nan" and u != "-"]
        unit_str = f" [{unit_vals[0]}]" if len(unit_vals) == 1 else ""

        ax.set_title(str(var))
        ax.set_xlabel("Time [h]")
        ax.set_ylabel(f"{var}{unit_str}")
        ax.legend(ncol=3, fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        plt.show()

def load_tidy_ambr(path: str) -> pd.DataFrame:
    """
    Load a tidy AMBR dataset (e.g., ambr_tidy_timeseries.csv) with
    robust dtypes for plotting.
    """
    df = pd.read_csv(path)

    # Coerce numeric columns
    if "Time" in df.columns:
        df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    if "BioreactorID" in df.columns:
        df["BioreactorID"] = pd.to_numeric(df["BioreactorID"], errors="coerce").astype("Int64")

    # Force text columns to strings (avoid float/NaN issues)
    for col in ("Run", "Variable", "Unit"):
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("")

    return df