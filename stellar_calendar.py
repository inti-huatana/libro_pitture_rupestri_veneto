#!/usr/bin/env python3
"""
stellar_calendar.py
Build a prehistoric stellar calendar from precomputed heliacal data (fullcat.csv).

Outputs (in ./output/):
  heliacal_events.csv      — all visible stars per epoch, sorted by rising day
  discontinuities.csv      — stars that appear/disappear or shift >30 d between millennia
  seasonal_calendar.csv    — which stars announced each seasonal window in each millennium
  epoch_summary.csv        — one-line digest per epoch
"""

from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_CSV = Path("fullcat.csv.gz") if Path("fullcat.csv.gz").exists() else Path("fullcat.csv")
OUTPUT_DIR = Path("output")

# Epochs to analyse: 0 to -100 kyr in 1-kyr steps
EPOCH_MIN = -100.0
EPOCH_MAX = 0.0

# Seasonal windows (days from spring equinox, 0–365).
# Overlapping by design — a star can announce multiple activities.
# Each tuple: (label_short, label_it, day_start, day_end)
SEASONS = [
    ("GELO",     "Gelo / nascite animali",          315, 365),
    ("GELO",     "Gelo / nascite animali",            0,  45),  # wraps through 0
    ("DISGELO",  "Disgelo / alluvioni alpine",        30,  75),
    ("SEMINA",   "Semina",                            60, 105),
    ("PASCOLO",  "Pascolo alpino / migrazioni su",    90, 150),
    ("CALURA",   "Calura estiva / raccolta erbe",    135, 195),
    ("RACCOLTO", "Raccolto / preparazione inverno",  180, 225),
    ("CACCIA_A", "Caccia autunnale / migrazioni giù",210, 270),
    ("FREDDO",   "Freddo / caccia invernale",        270, 315),
]

# Primary (non-overlapping) seasonal label per day — used for the main table
_SEASON_BINS = [0, 45, 90, 135, 180, 225, 270, 315, 365]
_SEASON_LABELS = [
    "Gelo/Inverno",        # 0–45
    "Disgelo/Primavera",   # 45–90
    "Semina/Pascolo",      # 90–135
    "Calura estiva",       # 135–180
    "Raccolto",            # 180–225
    "Caccia autunnale",    # 225–270
    "Freddo/Pre-inverno",  # 270–315
    "Gelo/Inverno",        # 315–365
]

SHIFT_THRESHOLD_DAYS = 30.0   # flag discontinuity if rising day shifts more than this

# --- Filter for heliacal_events and seasonal_calendar ---
# Keep stars with Vmag <= this threshold, PLUS all stars in SPECIAL_HIP regardless of magnitude.
VMAG_THRESHOLD = 2.0

# Asterisms to always include: Pleiades, Orion Belt, Southern Cross
SPECIAL_HIP: frozenset[int] = frozenset({
    # Pleiadi (members in Hp<4 catalog)
    17702,  # Alcyone   V=2.87
    17499,  # Electra   V=3.70
    17847,  # Atlas     V=3.62
    17573,  # Maia      V=3.88
    # Cintura di Orione
    25930,  # Mintaka   V=2.25
    26311,  # Alnilam   V=1.69  (also V<=2.0, listed for clarity)
    26727,  # Alnitak   V=1.77  (idem)
    # Croce del Sud
    60718,  # Acrux     V=0.78  (idem)
    62434,  # Mimosa    V=1.25  (idem)
    61084,  # Gacrux    V=1.62  (idem)
    59747,  # Imai      V=2.79
    60260,  # Ginan     V=3.59
})


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}")


def day_to_season(day_series: pd.Series) -> pd.Series:
    """Map a Series of days-from-equinox (0–365) to primary seasonal label."""
    day_mod = day_series.mod(365.0)
    return pd.cut(
        day_mod,
        bins=_SEASON_BINS,
        labels=_SEASON_LABELS,
        right=False,
        ordered=False,
    ).astype(str)


def star_label(row: pd.Series) -> str:
    """Return best available human-readable identifier for a star row."""
    if pd.notna(row.get("NAME")) and str(row["NAME"]).strip():
        return str(row["NAME"]).strip()
    if pd.notna(row.get("Bayer")) and str(row["Bayer"]).strip():
        return str(row["Bayer"]).strip()
    return f"HIP {int(row['HIP'])}"


def asterism_label(hip: int) -> str:
    """Return asterism tag for special HIPs, empty string otherwise."""
    pleiadi = {17702, 17499, 17847, 17573}
    orione  = {25930, 26311, 26727}
    crux    = {60718, 62434, 61084, 59747, 60260}
    if hip in pleiadi: return "Pleiadi"
    if hip in orione:  return "Cintura Orione"
    if hip in crux:    return "Croce del Sud"
    return ""


def filter_bright(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only V<=VMAG_THRESHOLD stars plus special asterism members."""
    mask = (df["Vmag"] <= VMAG_THRESHOLD) | df["HIP"].isin(SPECIAL_HIP)
    return df[mask].copy()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    log(f"Reading {path} ...")
    df = pd.read_csv(path, dtype={"HIP": "Int64", "HD": "float64"})
    log(f"  {len(df):,} rows raw, {df['HIP'].nunique()} stars, "
        f"{df['epoch_kyr_from_year0'].nunique()} epochs")

    # Filter to target epoch range
    mask = (df["epoch_kyr_from_year0"] >= EPOCH_MIN) & (df["epoch_kyr_from_year0"] <= EPOCH_MAX)
    df = df[mask].copy()

    # Round epoch to avoid float noise
    df["epoch_kyr"] = df["epoch_kyr_from_year0"].round(3)

    # Deduplicate at source: one row per (HIP, epoch_kyr)
    n_before = len(df)
    df = df.sort_values("epoch_kyr_from_year0", ascending=False)  # consistent tie-break
    df = df.drop_duplicates(subset=["HIP", "epoch_kyr"], keep="first").copy()
    n_after = len(df)
    if n_before != n_after:
        log(f"  Removed {n_before - n_after:,} duplicate (HIP, epoch) rows")

    log(f"  After filter+dedup: {n_after:,} rows "
        f"({df['HIP'].nunique()} stars × {df['epoch_kyr'].nunique()} epochs)")

    # Star label (best available name)
    df["star_label"] = df.apply(star_label, axis=1)

    return df


# ---------------------------------------------------------------------------
# Heliacal events table
# ---------------------------------------------------------------------------
def build_heliacal_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (epoch, star) where star is VISIBILE, filtered to bright+special.
    Sorted by epoch then rising day."""
    vis = df[df["visibility"] == "VISIBILE"].copy()
    vis = filter_bright(vis)

    vis["season_rising"]  = day_to_season(vis["heliacal_rising_day"])
    vis["season_setting"] = day_to_season(vis["heliacal_setting_day"])
    vis["asterismo"] = vis["HIP"].apply(lambda h: asterism_label(int(h)))

    cols = [
        "epoch_kyr", "HIP", "star_label", "asterismo", "Vmag",
        "heliacal_rising_day", "season_rising",
        "heliacal_setting_day", "season_setting",
        "acronychal_rising_day", "acronychal_setting_day",
        "dec_deg", "arcus_visionis_deg",
    ]
    out = vis[cols].sort_values(["epoch_kyr", "heliacal_rising_day"], ascending=[False, True])
    log(f"Heliacal table: {len(out):,} rows "
        f"(V<={VMAG_THRESHOLD} + {len(SPECIAL_HIP)} stelle speciali)")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Seasonal calendar pivot
# ---------------------------------------------------------------------------
def build_seasonal_calendar(heliacal: pd.DataFrame) -> pd.DataFrame:
    """
    For each epoch and each seasonal window, list stars (already filtered to
    bright+special) whose heliacal rising falls inside that window.
    Output: one row per (epoch, season_window, star), brightest first.
    """
    records = []
    for _, group in heliacal.groupby("epoch_kyr", sort=False):
        epoch  = group["epoch_kyr"].iloc[0]
        rising = group["heliacal_rising_day"].values
        labels = group["star_label"].values
        vmags  = group["Vmag"].values
        hips   = group["HIP"].values
        aster  = group["asterismo"].values

        for short, label_it, d_start, d_end in SEASONS:
            if d_start >= d_end:
                continue
            in_window = (rising >= d_start) & (rising < d_end)
            stars_in = [(hips[i], labels[i], vmags[i], rising[i], aster[i])
                        for i in np.where(in_window)[0]]
            stars_in.sort(key=lambda x: x[2])  # brightest first
            for hip, lbl, vmag, day, ast in stars_in:
                records.append({
                    "epoch_kyr":         epoch,
                    "window_short":      short,
                    "window_label":      label_it,
                    "day_start":         d_start,
                    "day_end":           d_end,
                    "HIP":               hip,
                    "star_label":        lbl,
                    "asterismo":         ast,
                    "Vmag":              round(vmag, 2),
                    "heliacal_rising_day": round(day, 1),
                })

    out = pd.DataFrame(records)
    log(f"Seasonal calendar: {len(out):,} rows")
    return out


# ---------------------------------------------------------------------------
# Discontinuity detection
# ---------------------------------------------------------------------------
def build_discontinuities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect, for each star, transitions between consecutive millennia:
      APPARE           — was invisible/absent, now visible
      SCOMPARE         — was visible, now invisible/absent
      DIVENTA_CIRCUMP  — was visible (had heliacal rising), now circumpolar
      CESSA_CIRCUMP    — was circumpolar, now has heliacal rising again
      SALTO_STAGIONALE — heliacal rising shifts >SHIFT_THRESHOLD_DAYS days

    Uses groupby+shift (fully vectorised) to avoid set_index/reindex fragility.
    """
    # Deduplicate: one row per (HIP, epoch_kyr), keep first
    d = (df.groupby(["HIP", "epoch_kyr"], sort=False)
           .first()
           .reset_index()
           .sort_values(["HIP", "epoch_kyr"], ascending=[True, False])  # newest first within HIP
           .copy())

    d["visibility"] = d["visibility"].fillna("ASSENTE")

    grp = d.groupby("HIP", sort=False)

    # "prev" = one millennium older (shift -1 because sorted newest→oldest)
    d["vis_prev"]    = grp["visibility"].shift(-1).fillna("ASSENTE")
    d["rise_prev"]   = grp["heliacal_rising_day"].shift(-1)
    d["epoch_prev"]  = grp["epoch_kyr"].shift(-1)

    # Drop rows with no previous epoch (last row per HIP)
    d = d.dropna(subset=["epoch_prev"]).copy()

    v_now  = d["visibility"]
    v_prev = d["vis_prev"]
    r_now  = d["heliacal_rising_day"]
    r_prev = d["rise_prev"]

    # Circular day-shift (wrap at 365)
    raw_delta = r_now - r_prev
    delta = np.where(raw_delta >  182.5, raw_delta - 365.0, raw_delta)
    delta = np.where(delta      < -182.5, delta    + 365.0, delta)
    d["_delta"] = delta

    masks = {
        "APPARE":               v_prev.isin(["INVISIBILE", "ASSENTE"]) & (v_now == "VISIBILE"),
        "SCOMPARE":             (v_prev == "VISIBILE") & v_now.isin(["INVISIBILE", "ASSENTE"]),
        "DIVENTA_CIRCUMPOLARE": (v_prev == "VISIBILE") & (v_now == "CIRCUMPOLARE"),
        "CESSA_CIRCUMPOLARE":   (v_prev == "CIRCUMPOLARE") & (v_now == "VISIBILE"),
        "SALTO_STAGIONALE":     (v_now == "VISIBILE") & (v_prev == "VISIBILE")
                                & r_now.notna() & r_prev.notna()
                                & (d["_delta"].abs() >= SHIFT_THRESHOLD_DAYS),
    }

    def det(row: pd.Series, evento: str) -> str:
        if evento == "APPARE":
            return f"giorno levata: {row['heliacal_rising_day']:.1f}" if pd.notna(row["heliacal_rising_day"]) else ""
        if evento == "SCOMPARE":
            return f"ultimo giorno levata: {row['rise_prev']:.1f}" if pd.notna(row["rise_prev"]) else ""
        if evento == "SALTO_STAGIONALE":
            return f"Δ={row['_delta']:+.1f}d ({row['rise_prev']:.1f}→{row['heliacal_rising_day']:.1f})"
        return ""

    parts: list[pd.DataFrame] = []
    for evento, mask in masks.items():
        sub = d[mask].copy()
        sub["evento"]     = evento
        sub["dettaglio"]  = sub.apply(det, axis=1, evento=evento)
        sub["epoch_da_kyr"] = sub["epoch_prev"]
        sub["epoch_a_kyr"]  = sub["epoch_kyr"]
        parts.append(sub[["epoch_da_kyr", "epoch_a_kyr", "HIP", "star_label", "Vmag",
                           "evento", "dettaglio"]])

    out = (pd.concat(parts, ignore_index=True)
             .sort_values(["epoch_a_kyr", "Vmag", "star_label"], ascending=[False, True, True])
             .reset_index(drop=True))
    log(f"Discontinuities: {len(out):,} events")
    return out


# ---------------------------------------------------------------------------
# Epoch summary
# ---------------------------------------------------------------------------
def build_epoch_summary(df: pd.DataFrame, heliacal: pd.DataFrame) -> pd.DataFrame:
    records = []
    epochs = sorted(df["epoch_kyr"].unique(), reverse=True)

    for epoch in epochs:
        all_epoch = df[df["epoch_kyr"] == epoch]
        vis_epoch = heliacal[heliacal["epoch_kyr"] == epoch]

        n_vis   = len(vis_epoch)
        n_invis = (all_epoch["visibility"] == "INVISIBILE").sum()
        n_circ  = (all_epoch["visibility"] == "CIRCUMPOLARE").sum() if "CIRCUMPOLARE" in all_epoch["visibility"].values else 0

        if n_vis > 0:
            earliest = vis_epoch.loc[vis_epoch["heliacal_rising_day"].idxmin()]
            latest   = vis_epoch.loc[vis_epoch["heliacal_rising_day"].idxmax()]

            # Longest gap between consecutive rising days (sorted)
            sorted_days = np.sort(vis_epoch["heliacal_rising_day"].dropna().values)
            if len(sorted_days) > 1:
                gaps = np.diff(sorted_days)
                wrap_gap = 365.0 - sorted_days[-1] + sorted_days[0]
                all_gaps = np.append(gaps, wrap_gap)
                max_gap_idx = np.argmax(all_gaps)
                if max_gap_idx < len(gaps):
                    gap_start = sorted_days[max_gap_idx]
                    gap_end   = sorted_days[max_gap_idx + 1]
                else:
                    gap_start = sorted_days[-1]
                    gap_end   = sorted_days[0] + 365.0
                max_gap = float(all_gaps[max_gap_idx])
            else:
                gap_start = gap_end = max_gap = np.nan

            records.append({
                "epoch_kyr":           epoch,
                "n_visibili":          n_vis,
                "n_invisibili":        n_invis,
                "n_circumpolari":      n_circ,
                "levata_piu_precoce":  f"{earliest['star_label']} g.{earliest['heliacal_rising_day']:.0f}",
                "levata_piu_tardiva":  f"{latest['star_label']} g.{latest['heliacal_rising_day']:.0f}",
                "vuoto_max_giorni":    round(max_gap, 1) if not np.isnan(max_gap) else None,
                "vuoto_da_giorno":     round(gap_start, 1) if not np.isnan(gap_start) else None,
                "vuoto_a_giorno":      round(gap_end % 365.0, 1) if not np.isnan(gap_end) else None,
            })
        else:
            records.append({
                "epoch_kyr": epoch, "n_visibili": 0,
                "n_invisibili": n_invis, "n_circumpolari": n_circ,
                "levata_piu_precoce": "", "levata_piu_tardiva": "",
                "vuoto_max_giorni": None, "vuoto_da_giorno": None, "vuoto_a_giorno": None,
            })

    out = pd.DataFrame(records)
    log(f"Epoch summary: {len(out)} epochs")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_data(INPUT_CSV)

    log("Building heliacal events table ...")
    heliacal = build_heliacal_table(df)
    out_path = OUTPUT_DIR / "heliacal_events.csv"
    heliacal.to_csv(out_path, index=False, float_format="%.2f")
    log(f"  → {out_path}")

    log("Building seasonal calendar ...")
    seasonal = build_seasonal_calendar(heliacal)
    out_path = OUTPUT_DIR / "seasonal_calendar.csv"
    seasonal.to_csv(out_path, index=False, float_format="%.2f")
    log(f"  → {out_path}")

    log("Detecting discontinuities ...")
    disc = build_discontinuities(df)
    out_path = OUTPUT_DIR / "discontinuities.csv"
    disc.to_csv(out_path, index=False, float_format="%.2f")
    log(f"  → {out_path}")

    log("Building epoch summary ...")
    summary = build_epoch_summary(df, heliacal)
    out_path = OUTPUT_DIR / "epoch_summary.csv"
    summary.to_csv(out_path, index=False, float_format="%.2f")
    log(f"  → {out_path}")

    # Quick sanity print
    print("\n--- Esempi discontinuità (prime 20) ---")
    print(disc[["epoch_a_kyr", "star_label", "Vmag", "evento", "dettaglio"]].head(20).to_string(index=False))

    print("\n--- Riepilogo epoche (prime 10) ---")
    print(summary.head(10).to_string(index=False))

    log("Done.")


if __name__ == "__main__":
    main()
