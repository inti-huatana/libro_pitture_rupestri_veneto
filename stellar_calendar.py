#!/usr/bin/env python3
"""
stellar_calendar.py
Process fullcat.csv.gz into prehistoric stellar calendar tables.

Usage:
    python3 stellar_calendar.py [options]

Options:
    --input     Path to fullcat.csv or fullcat.csv.gz  [default: auto-detect]
    --outdir    Output directory                        [default: output]
    --lat       Observer latitude in degrees            [default: 45.0]
    --Tmin      Minimum epoch in kyr (negative = past)  [default: -100.0]
    --Tmax      Maximum epoch in kyr                    [default: 0.0]
    --dt        Timestep in kyr                         [default: 1.0]
"""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd

# ── Asterism representatives (4 stars only) ─────────────────────────────────
ASTERISM_HIP: dict[int, str] = {
    17702: "Pleiadi",
    26727: "Cin.Orione",
    60718: "Cr.Sud",
    20894: "Iadi",
}

# ── Narrow seasonal windows: (name, center_day, half_width) ─────────────────
# center_day = days from vernal equinox (day 0). Window = [center-hw, center+hw].
SEASONAL_WINDOWS: list[tuple[str, float, float]] = [
    ("Equinozio primavera",   0.0,   5.0),
    ("Inizio semina",        46.0,   5.0),
    ("Solstizio estivo",     92.0,   5.0),
    ("Inizio raccolta",     137.0,   5.0),
    ("Equinozio autunno",   183.0,   5.0),
    ("Fine raccolta",       228.0,   5.0),
    ("Solstizio invernale", 274.0,   5.0),
    ("Inizio freddo",       319.0,   5.0),
]


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prehistoric stellar calendar from fullcat.csv.gz")
    p.add_argument("--input",  default=None,     help="Input CSV path")
    p.add_argument("--outdir", default="output",  help="Output directory")
    p.add_argument("--lat",    type=float, default=45.0,   help="Observer latitude in degrees")
    p.add_argument("--Tmin",   type=float, default=-100.0, help="Min epoch in kyr")
    p.add_argument("--Tmax",   type=float, default=0.0,    help="Max epoch in kyr")
    p.add_argument("--dt",     type=float, default=1.0,    help="Timestep in kyr")
    return p.parse_args()


def _star_label(row) -> str:
    name  = str(row.get("NAME",  "")).strip()
    bayer = str(row.get("Bayer", "")).strip()
    if name and name != "nan":
        label = name
    elif bayer and bayer != "nan":
        label = bayer
    else:
        label = f"HIP {int(row['HIP'])}"
    hip = int(row["HIP"])
    if hip in ASTERISM_HIP:
        label += f" ({ASTERISM_HIP[hip]})"
    return label


def _season_label(day) -> str:
    try:
        d = float(day) % 365.2422
    except (TypeError, ValueError):
        return ""
    for name, center, hw in SEASONAL_WINDOWS:
        lo = (center - hw) % 365.2422
        hi = (center + hw) % 365.2422
        if lo <= hi:
            if lo <= d <= hi:
                return name
        else:
            if d >= lo or d <= hi:
                return name
    return ""


def load_data(input_path: Path, lat_deg: float,
              tmin: float, tmax: float, dt: float) -> pd.DataFrame:
    log(f"Loading {input_path} ...")
    df = pd.read_csv(input_path, low_memory=False)
    log(f"  Raw rows: {len(df):,}")

    if "epoch_kyr_from_year0" in df.columns and "epoch_kyr" not in df.columns:
        df = df.rename(columns={"epoch_kyr_from_year0": "epoch_kyr"})

    df = df.drop_duplicates(subset=["HIP", "epoch_kyr"], keep="first")
    log(f"  After dedup: {len(df):,}")

    # Select epochs matching tmin..tmax at step dt
    n_steps = round((tmax - tmin) / dt)
    target_epochs = {round(tmin + i * dt, 6) for i in range(n_steps + 1)}
    df["_er"] = df["epoch_kyr"].round(6)
    df = df[df["_er"].isin(target_epochs)].drop(columns=["_er"])
    log(f"  After epoch filter [{tmin}, {tmax}] step {dt}: "
        f"{len(df):,} rows, {df['epoch_kyr'].nunique()} epochs")

    # Meridian altitude filter: 90 - |lat - dec| >= 2
    df = df.dropna(subset=["dec_deg"])
    alt = 90.0 - (lat_deg - df["dec_deg"]).abs()
    df = df[alt >= 2.0].copy()
    log(f"  After meridian alt >=2° filter: {len(df):,} rows")

    df["star_label"] = df.apply(_star_label, axis=1)
    return df.reset_index(drop=True)


def build_heliacal_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per epoch: top-2 brightest visible + 1 per asterism (if visible)."""
    vis = df[
        (df["visibility"] == "VISIBILE") &
        df["heliacal_rising_day"].notna()
    ].copy()
    vis["asterism"] = vis["HIP"].map(ASTERISM_HIP).fillna("")

    records: list[dict] = []
    for epoch, grp in vis.groupby("epoch_kyr", sort=True):
        grp_s = grp.sort_values("Vmag")
        top2 = grp_s.head(2)
        selected = set(top2["HIP"].tolist())

        extra: list[pd.Series] = []
        for hip in ASTERISM_HIP:
            if hip not in selected:
                rows = grp_s[grp_s["HIP"] == hip]
                if not rows.empty:
                    extra.append(rows.iloc[0])
                    selected.add(hip)

        chosen = pd.concat([top2] + ([pd.DataFrame(extra)] if extra else []))
        for _, row in chosen.iterrows():
            records.append({
                "epoch_kyr":            row["epoch_kyr"],
                "HIP":                  int(row["HIP"]),
                "star_label":           row["star_label"],
                "asterism":             row["asterism"],
                "Vmag":                 row["Vmag"],
                "heliacal_rising_day":  row["heliacal_rising_day"],
                "heliacal_setting_day": row.get("heliacal_setting_day", float("nan")),
                "season_rising":        _season_label(row["heliacal_rising_day"]),
                "season_setting":       _season_label(row.get("heliacal_setting_day", float("nan"))),
            })

    out = pd.DataFrame(records)
    if not out.empty:
        out = out.sort_values(["epoch_kyr", "Vmag"])
    log(f"  Heliacal events: {len(out):,} rows")
    return out


def build_seasonal_calendar(heliacal: pd.DataFrame) -> pd.DataFrame:
    """Stars whose heliacal rising falls within a narrow seasonal window."""
    if heliacal.empty:
        return pd.DataFrame()
    sub = heliacal[heliacal["season_rising"] != ""].copy()
    out = sub[["epoch_kyr", "season_rising", "star_label", "asterism",
               "Vmag", "heliacal_rising_day"]].rename(
        columns={"season_rising": "season", "heliacal_rising_day": "rising_day"})
    out = out.sort_values(["epoch_kyr", "season", "Vmag"])
    log(f"  Seasonal calendar: {len(out):,} rows")
    return out


def build_discontinuities(df: pd.DataFrame) -> pd.DataFrame:
    """Detect appearances, disappearances, circumpolar transitions."""
    d = (df.groupby(["HIP", "epoch_kyr"], sort=False)
           .first().reset_index()
           .sort_values(["HIP", "epoch_kyr"], ascending=[True, False])
           .copy())
    d["visibility"] = d["visibility"].fillna("ASSENTE")

    grp = d.groupby("HIP", sort=False)
    d["vis_prev"]   = grp["visibility"].shift(-1).fillna("ASSENTE")
    d["rise_prev"]  = grp["heliacal_rising_day"].shift(-1)
    d["epoch_prev"] = grp["epoch_kyr"].shift(-1)
    d = d.dropna(subset=["epoch_prev"]).copy()

    vis  = d["visibility"]
    visp = d["vis_prev"]

    event_masks: dict[str, pd.Series] = {
        "APPARE":               (visp == "ASSENTE") & (vis == "VISIBILE"),
        "SCOMPARE":             (visp == "VISIBILE") & (vis == "ASSENTE"),
        "DIVENTA_CIRCUMPOLARE": (visp != "CIRCUMPOLARE") & (vis == "CIRCUMPOLARE"),
        "CESSA_CIRCUMPOLARE":   (visp == "CIRCUMPOLARE") & (vis != "CIRCUMPOLARE"),
        "SALTO_STAGIONALE": (
            (vis == "VISIBILE") & (visp == "VISIBILE") &
            d["heliacal_rising_day"].notna() & d["rise_prev"].notna() &
            ((d["heliacal_rising_day"] - d["rise_prev"]).abs() > 30)
        ),
    }

    parts: list[pd.DataFrame] = []
    for evento, mask in event_masks.items():
        sub = d[mask].copy()
        sub["evento"]      = evento
        sub["epoch_a_kyr"] = sub["epoch_kyr"]
        if evento == "SALTO_STAGIONALE":
            sub["dettaglio"] = (
                "g." + sub["heliacal_rising_day"].round(0).astype(int).astype(str) +
                " <- g." + sub["rise_prev"].round(0).astype(int).astype(str)
            )
        else:
            sub["dettaglio"] = ""
        parts.append(sub[["epoch_a_kyr", "HIP", "star_label", "Vmag", "evento", "dettaglio"]])

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["epoch_a_kyr", "Vmag"], ascending=[False, True])
    log(f"  Discontinuities: {len(out):,} rows")
    return out


def build_pole_star_table(df: pd.DataFrame) -> pd.DataFrame:
    """Best pole star per epoch: score = pole_distance_deg + 3.0 * Vmag."""
    circ = df[df["visibility"] == "CIRCUMPOLARE"].dropna(subset=["dec_deg", "Vmag"]).copy()
    circ["pole_dist"] = 90.0 - circ["dec_deg"]
    circ = circ[circ["pole_dist"] >= 0]
    circ["score"] = circ["pole_dist"] + 3.0 * circ["Vmag"]

    records: list[dict] = []
    for epoch, grp in circ.groupby("epoch_kyr", sort=True):
        idx = grp["score"].idxmin()
        best = grp.loc[idx]
        records.append({
            "epoch_kyr":     epoch,
            "HIP":           int(best["HIP"]),
            "star_label":    best["star_label"],
            "Vmag":          best["Vmag"],
            "dec_deg":       best["dec_deg"],
            "pole_dist_deg": best["pole_dist"],
            "score":         best["score"],
        })

    out = pd.DataFrame(records)
    log(f"  Pole star table: {len(out):,} epochs")
    return out


def build_epoch_summary(df: pd.DataFrame, heliacal: pd.DataFrame) -> pd.DataFrame:
    vis_count  = (df[df["visibility"] == "VISIBILE"]
                  .groupby("epoch_kyr")["HIP"].nunique()
                  .rename("n_visible"))
    circ_count = (df[df["visibility"] == "CIRCUMPOLARE"]
                  .groupby("epoch_kyr")["HIP"].nunique()
                  .rename("n_circumpolare"))
    hel_count  = (heliacal.groupby("epoch_kyr")["HIP"].nunique()
                  .rename("n_heliacal") if not heliacal.empty
                  else pd.Series(dtype=int, name="n_heliacal"))

    out = (pd.concat([vis_count, circ_count, hel_count], axis=1)
             .fillna(0).astype(int)
             .reset_index()
             .sort_values("epoch_kyr", ascending=False))
    log(f"  Epoch summary: {len(out):,} epochs")
    return out


def main() -> None:
    args = parse_args()

    if args.input:
        input_path = Path(args.input)
    else:
        input_path = (Path("fullcat.csv.gz") if Path("fullcat.csv.gz").exists()
                      else Path("fullcat.csv"))

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path, args.lat, args.Tmin, args.Tmax, args.dt)

    log("Building heliacal table ...")
    heliacal = build_heliacal_table(df)

    log("Building seasonal calendar ...")
    seasonal = build_seasonal_calendar(heliacal)

    log("Building discontinuities ...")
    disc = build_discontinuities(df)

    log("Building pole star table ...")
    poles = build_pole_star_table(df)

    log("Building epoch summary ...")
    summary = build_epoch_summary(df, heliacal)

    heliacal.to_csv(out_dir / "heliacal_events.csv",   index=False)
    seasonal.to_csv(out_dir / "seasonal_calendar.csv",  index=False)
    disc.to_csv(    out_dir / "discontinuities.csv",    index=False)
    poles.to_csv(   out_dir / "pole_stars.csv",         index=False)
    summary.to_csv( out_dir / "epoch_summary.csv",      index=False)

    log(f"Done. Output in {out_dir}/")
    log(f"  heliacal_events.csv    {len(heliacal):,} rows")
    log(f"  seasonal_calendar.csv  {len(seasonal):,} rows")
    log(f"  discontinuities.csv    {len(disc):,} rows")
    log(f"  pole_stars.csv         {len(poles):,} rows")
    log(f"  epoch_summary.csv      {len(summary):,} rows")


if __name__ == "__main__":
    main()
