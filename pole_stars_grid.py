#!/usr/bin/env python3
"""
pole_stars_grid.py
Traccia le N stelle più vicine al polo nord e al polo sud per ogni timestep,
usando i singoli HIP_NNN.csv prodotti da bright_star_longterm.py.

La declinazione di una stella (dec_deg) non dipende dalla latitudine
dell'osservatore: è una proprietà celeste che varia solo per precessione e
moto proprio. Si legge perciò da un'unica directory di riferimento.

Usage:
    python3 pole_stars_grid.py --refdir BS45N [--outdir output_poles] [--n 10]
    python3 pole_stars_grid.py --refdir BS45N --refdir-s BS40S --n 10

Output:
    <outdir>/pole_north_top<n>.csv
    <outdir>/pole_south_top<n>.csv
    <outdir>/pole_north_top<n>.pdf
    <outdir>/pole_south_top<n>.pdf
"""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ── helpers ──────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}")


def star_label(row: pd.Series) -> str:
    name  = str(row.get("NAME",  "")).strip()
    bayer = str(row.get("Bayer", "")).strip()
    if name  and name  != "nan": return name
    if bayer and bayer != "nan": return bayer
    return f"HIP {int(row['HIP'])}"


def read_one(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, low_memory=False,
                         usecols=lambda c: c in {
                             "epoch_kyr_from_year0", "epoch_kyr",
                             "HIP", "NAME", "Bayer", "Vmag",
                             "dec_deg", "visibility",
                         })
        if "epoch_kyr_from_year0" in df.columns and "epoch_kyr" not in df.columns:
            df = df.rename(columns={"epoch_kyr_from_year0": "epoch_kyr"})
        return df
    except Exception as e:
        log(f"  SKIP {path.name}: {e}")
        return None


def load_catalog(ref_dir: Path, n_workers: int = 8) -> pd.DataFrame:
    """Legge tutti HIP_*.csv da ref_dir in parallelo; restituisce DataFrame."""
    hip_files = sorted(ref_dir.glob("HIP_*.csv"))
    log(f"  {len(hip_files)} file HIP in {ref_dir}")
    chunks: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(read_one, p): p for p in hip_files}
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            if res is not None:
                chunks.append(res)
            if i % 100 == 0:
                log(f"    letti {i}/{len(hip_files)}")
    df = pd.concat(chunks, ignore_index=True)
    log(f"  Totale: {len(df):,} righe, {df['HIP'].nunique()} stelle, "
        f"{df['epoch_kyr'].nunique()} epoche")
    return df


# ── analisi poli ─────────────────────────────────────────────────────────────

def top_n_per_epoch(df: pd.DataFrame, pole: str, n: int) -> pd.DataFrame:
    """
    Restituisce le n stelle per distanza minima dal polo richiesto,
    per ogni epoch_kyr.

    pole: 'N' (NCP, 90 - dec_deg) oppure 'S' (SCP, dec_deg + 90)
    """
    d = df.dropna(subset=["dec_deg", "Vmag"]).copy()
    if pole == "N":
        d["pole_dist_deg"] = 90.0 - d["dec_deg"]
    else:
        d["pole_dist_deg"] = d["dec_deg"] + 90.0

    # Per il polo nord vogliamo dec più alta → dist più bassa
    # Per il polo sud vogliamo dec più bassa → dist più bassa
    # In entrambi i casi: teniamo pole_dist_deg minima (>=0)
    d = d[d["pole_dist_deg"] >= 0.0]

    records: list[dict] = []
    for epoch, grp in d.groupby("epoch_kyr", sort=True):
        top = grp.nsmallest(n, "pole_dist_deg")
        for rank, (_, row) in enumerate(top.iterrows(), 1):
            records.append({
                "epoch_kyr":    epoch,
                "rank":         rank,
                "HIP":          int(row["HIP"]),
                "label":        star_label(row),
                "Vmag":         row["Vmag"],
                "dec_deg":      row["dec_deg"],
                "pole_dist_deg": row["pole_dist_deg"],
                "visibility":   row.get("visibility", ""),
            })

    return pd.DataFrame(records)


# ── plotting ──────────────────────────────────────────────────────────────────

def _star_colors(labels: list[str]) -> dict[str, str]:
    cmap = plt.get_cmap("tab20", len(labels))
    return {lbl: matplotlib.colors.to_hex(cmap(i)) for i, lbl in enumerate(labels)}


def plot_poles(top_df: pd.DataFrame, full_df: pd.DataFrame,
               pole: str, n: int, out_path: Path) -> None:
    """
    Due pannelli:
      In alto  — distanza dal polo vs. epoca per le stelle che compaiono
                 almeno una volta in top-n (linea intera); pallino pieno
                 quando effettivamente in top-n.
      In basso — magnitudine apparente Vmag vs. epoca delle stesse stelle.
    """
    pole_col = "pole_dist_deg"
    pole_label = "NCP" if pole == "N" else "SCP"

    # Stelle che compaiono almeno una volta nel top-n
    ever_top = sorted(top_df["label"].unique())
    colors = _star_colors(ever_top)

    # Serie complete per queste stelle dal catalogo completo
    if pole == "N":
        full_df = full_df.copy()
        full_df[pole_col] = 90.0 - full_df["dec_deg"]
    else:
        full_df = full_df.copy()
        full_df[pole_col] = full_df["dec_deg"] + 90.0
    full_df["label"] = full_df.apply(star_label, axis=1)
    full_sub = full_df[full_df["label"].isin(ever_top)].copy()

    epochs = np.sort(top_df["epoch_kyr"].unique())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})

    for lbl in ever_top:
        c = colors[lbl]
        # Linea completa (full time range)
        star_full = full_sub[full_sub["label"] == lbl].sort_values("epoch_kyr")
        ax1.plot(star_full["epoch_kyr"], star_full[pole_col],
                 color=c, lw=0.8, alpha=0.6)
        ax2.plot(star_full["epoch_kyr"], star_full["Vmag"],
                 color=c, lw=0.8, alpha=0.6)

        # Pallini dove è in top-n
        in_top = top_df[top_df["label"] == lbl]
        ax1.scatter(in_top["epoch_kyr"], in_top[pole_col],
                    color=c, s=12, zorder=3)

    ax1.set_ylabel(f"Distanza angolare da {pole_label} [°]", fontsize=11)
    ax1.set_ylim(bottom=0)
    ax1.axhline(1, color="k", lw=0.5, ls="--", alpha=0.4)
    ax1.axhline(5, color="k", lw=0.5, ls=":",  alpha=0.4)
    ax1.set_title(f"Top-{n} stelle più vicine al polo {'nord' if pole=='N' else 'sud'} "
                  f"(0 – {top_df['epoch_kyr'].nunique()} timestep)", fontsize=13)
    ax1.grid(True, alpha=0.25)

    ax2.set_ylabel("Vmag", fontsize=11)
    ax2.set_xlabel("Epoca [kyr da anno 0]", fontsize=11)
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.25)

    # Legenda
    legend_handles = [
        Line2D([0], [0], color=colors[lbl], lw=1.5, label=lbl)
        for lbl in ever_top
    ]
    ax1.legend(handles=legend_handles, fontsize=7, ncol=2,
               loc="upper right", framealpha=0.85)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Salvato → {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Top-N stelle vicine ai poli celesti per ogni timestep")
    p.add_argument("--refdir",   default="BS40N",
                   help="Directory con HIP_*.csv per polo nord [default: BS40N]")
    p.add_argument("--refdir-s", default="BS40S",
                   help="Directory con HIP_*.csv per polo sud [default: BS40S]")
    p.add_argument("--outdir",   default="output_poles",
                   help="Directory di output [default: output_poles]")
    p.add_argument("--n",        type=int, default=10,
                   help="Numero di stelle per polo [default: 10]")
    p.add_argument("--workers",  type=int, default=8,
                   help="Thread per la lettura parallela [default: 8]")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ref_n = Path(args.refdir)
    ref_s = Path(args.refdir_s) if args.refdir_s else ref_n
    out   = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    n = args.n

    # ── polo nord ──
    log(f"Carico catalogo per polo nord da {ref_n} ...")
    df_n = load_catalog(ref_n, args.workers)
    log("Calcolo top-N polo nord ...")
    top_n = top_n_per_epoch(df_n, pole="N", n=n)
    csv_n = out / f"pole_north_top{n}.csv"
    top_n.to_csv(csv_n, index=False)
    log(f"  Salvato → {csv_n}")
    log("Plotting polo nord ...")
    plot_poles(top_n, df_n, pole="N", n=n,
               out_path=out / f"pole_north_top{n}.pdf")

    # ── polo sud ──
    if ref_s != ref_n:
        log(f"Carico catalogo per polo sud da {ref_s} ...")
        df_s = load_catalog(ref_s, args.workers)
    else:
        df_s = df_n
    log("Calcolo top-N polo sud ...")
    top_s = top_n_per_epoch(df_s, pole="S", n=n)
    csv_s = out / f"pole_south_top{n}.csv"
    top_s.to_csv(csv_s, index=False)
    log(f"  Salvato → {csv_s}")
    log("Plotting polo sud ...")
    plot_poles(top_s, df_s, pole="S", n=n,
               out_path=out / f"pole_south_top{n}.pdf")

    log("=== Fine ===")
    log(f"Output in {out}/")
    log(f"  {csv_n.name}  —  {len(top_n):,} righe")
    log(f"  {csv_s.name}  —  {len(top_s):,} righe")


if __name__ == "__main__":
    main()
