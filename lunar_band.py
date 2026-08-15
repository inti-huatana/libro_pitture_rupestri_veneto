#!/usr/bin/env python3
"""
lunar_band.py

Which bright stars the Moon could hide, epoch by epoch.

The Moon stays within 5.145 degrees of the ecliptic, and its node goes round in
18.6 years, so over that cycle it sweeps the whole band. Adding its parallax,
which lifts or lowers it by up to about a degree for an observer on the surface,
and its own semidiameter, the strip within which an occultation can happen from
somewhere on Earth reaches about 6.4 degrees of ecliptic latitude.

So the question is an inequality on a column the trajectory file already
carries. No ephemeris is needed: it does not ask where the Moon was on a given
night, which tidal dissipation makes unknowable this far back, but which part of
the sky it could reach, which is slow geometry.

An occultation of a first-magnitude star is the most striking thing the naked
eye ever sees happen: the star does not fade, it is cut off against a hard edge
and returns an hour later. Today the band holds exactly the classical four --
Aldebaran, Regulus, Spica, Antares -- and the Pleiades. But proper motion moves
stars through it: Aldebaran shifts by some eleven degrees over two hundred
millennia and Regulus by more, while Spica and Antares barely move. So the set
is not a constant of the sky, and when each star entered and left it is what
this computes.

Usage
-----
    python3 lunar_band.py --trajectories BSGRID/trajectories.csv --outdir moon
"""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd

from star_table import (normalise_epoch, pivot_epoch_star,
                        read_star_table, star_labels)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Inclination of the lunar orbit, degrees: the band the node precession sweeps.
LUNAR_INCLINATION_DEG = 5.145

# Equatorial horizontal parallax, about 0.95 degrees at mean distance, plus the
# semidiameter of some 0.26. Together they widen the strip within which an
# occultation is possible from somewhere on Earth.
PARALLAX_DEG = 0.95
SEMIDIAMETER_DEG = 0.26

BAND_TOTAL_DEG = LUNAR_INCLINATION_DEG + PARALLAX_DEG + SEMIDIAMETER_DEG
BAND_CENTRAL_DEG = LUNAR_INCLINATION_DEG          # occultable from most of Earth


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


def spells(epochs: np.ndarray, inside: np.ndarray, min_span: float):
    """Contiguous stretches during which a star stays inside the band."""
    out = []
    idx = np.flatnonzero(inside)
    if idx.size == 0:
        return out
    for seg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        lo, hi = min(epochs[seg[0]], epochs[seg[-1]]), max(epochs[seg[0]], epochs[seg[-1]])
        if hi - lo >= min_span:
            out.append((lo, hi, seg))
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Bright stars within reach of lunar occultation, through time.")
    p.add_argument("--trajectories", required=True)
    p.add_argument("--outdir", default="moon")
    p.add_argument("--vmax", type=float, default=3.0,
                   help="brightest magnitude considered; an occultation only "
                        "counts as an event if the star is conspicuous")
    p.add_argument("--min-span", type=float, default=1.0,
                   help="shortest spell reported, kyr")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log(f"Leggo {args.trajectories} ...")
    traj = read_star_table(args.trajectories,
                           ["epoch_kyr_from_year0", "HIP", "ecl_lat_deg",
                            "ecl_lon_deg", "Vmag", "NAME", "Bayer"])
    traj = normalise_epoch(traj).drop_duplicates(["HIP", "epoch"])

    labels = star_labels(traj)
    epochs, hips, _arr = pivot_epoch_star(traj, ["ecl_lat_deg", "Vmag"])
    beta, mag = _arr["ecl_lat_deg"], _arr["Vmag"]
    log(f"  {hips.size} stelle, {epochs.size} epoche "
        f"[{epochs.min():+.1f}, {epochs.max():+.1f}] kyr")
    log(f"  fascia: |lat. eclittica| < {BAND_CENTRAL_DEG:.2f}° centrale, "
        f"{BAND_TOTAL_DEG:.2f}° con parallasse e semidiametro")

    bright = mag <= args.vmax
    inside_total = bright & (np.abs(beta) <= BAND_TOTAL_DEG)
    inside_central = bright & (np.abs(beta) <= BAND_CENTRAL_DEG)

    # ---- per-star spells --------------------------------------------------
    rows = []
    for k in range(hips.size):
        if not inside_total[:, k].any():
            continue
        for lo, hi, seg in spells(epochs, inside_total[:, k], args.min_span):
            j = seg[int(np.argmin(np.abs(beta[seg, k])))]
            rows.append({
                "HIP": int(hips[k]), "star": labels.get(int(hips[k]), ""),
                "Vmag_now": float(mag[-1, k]),
                "epoch_start_kyr": float(lo), "epoch_end_kyr": float(hi),
                "duration_kyr": float(hi - lo),
                "ecl_lat_start_deg": float(beta[seg[0], k]),
                "ecl_lat_end_deg": float(beta[seg[-1], k]),
                "closest_ecl_lat_deg": float(beta[j, k]),
                "epoch_closest_kyr": float(epochs[j]),
                "central_ever": bool(inside_central[seg, k].any()),
            })
    sp = pd.DataFrame(rows)
    if sp.empty:
        raise SystemExit("Nessuna stella entro la fascia con questi limiti.")
    sp = sp.sort_values(["star", "epoch_start_kyr"])
    sp.to_csv(outdir / "spells.csv", index=False, float_format="%.4f")

    # ---- how many at once -------------------------------------------------
    count = pd.DataFrame({
        "epoch_kyr": epochs,
        "n_occultable_total": inside_total.sum(axis=1),
        "n_occultable_central": inside_central.sum(axis=1),
    })
    count.to_csv(outdir / "count.csv", index=False)

    now = int(np.argmin(np.abs(epochs - epochs.max())))
    log("")
    log(f"  Occultabili adesso ({epochs[now]:+.1f} kyr), V<{args.vmax}: "
        f"{int(inside_total[now].sum())}")
    for k in np.flatnonzero(inside_total[now]):
        log(f"    {labels.get(int(hips[k]), ''):16s} "
            f"lat.ecl {beta[now, k]:+6.2f}°  V {mag[now, k]:4.1f}")

    log("")
    log(f"  Numero di stelle occultabili nel tempo: da "
        f"{count['n_occultable_total'].min()} a "
        f"{count['n_occultable_total'].max()}")

    log("")
    log("  Periodi di occultabilita', per durata:")
    for _, r in sp.sort_values("duration_kyr", ascending=False).head(15).iterrows():
        flag = "" if r["central_ever"] else "  (solo di striscio)"
        log(f"    {r['star']:16s} {r['epoch_start_kyr']:+7.1f} .. "
            f"{r['epoch_end_kyr']:+7.1f} kyr ({r['duration_kyr']:6.1f}) | "
            f"minimo |b| {abs(r['closest_ecl_lat_deg']):4.2f}° a "
            f"{r['epoch_closest_kyr']:+7.1f}{flag}")

    # ---- figures -----------------------------------------------------------
    ever = np.flatnonzero(inside_total.any(axis=0))
    order = ever[np.argsort([np.nanmin(np.abs(beta[:, k])) for k in ever])]

    fig, ax = plt.subplots(figsize=(12, max(5, 0.32 * order.size)))
    for y, k in enumerate(order):
        ax.plot(epochs, np.full(epochs.size, y), color="0.9", lw=0.8, zorder=1)
        m = inside_total[:, k]
        ax.scatter(epochs[m], np.full(m.sum(), y), s=4,
                   c=np.abs(beta[m, k]), cmap="viridis_r",
                   vmin=0, vmax=BAND_TOTAL_DEG, zorder=2)
    ax.set_yticks(range(order.size))
    ax.set_yticklabels([f"{labels.get(int(hips[k]), '')} "
                        f"(V{mag[-1, k]:.1f})" for k in order], fontsize=7)
    ax.set_xlabel("Epoca [kyr dall'anno 0]")
    ax.set_title(f"Stelle V<{args.vmax} che la Luna poteva nascondere\n"
                 f"colore: distanza dal centro della fascia, scuro = piu' "
                 f"centrale e occultato piu' spesso")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "spells.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for k in order:
        a1.plot(epochs, beta[:, k], lw=1.0,
                label=labels.get(int(hips[k]), ""))
    for lim, st, lb in ((BAND_CENTRAL_DEG, "-", "fascia centrale"),
                        (BAND_TOTAL_DEG, "--", "con parallasse")):
        a1.axhline(lim, color="k", lw=0.8, ls=st)
        a1.axhline(-lim, color="k", lw=0.8, ls=st, label=lb)
    a1.set_ylabel("latitudine eclittica [°]")
    a1.set_ylim(-15, 15)
    a1.legend(fontsize=6, ncol=4)
    a1.grid(alpha=0.3)
    a1.set_title("Passaggio delle stelle attraverso la fascia lunare")

    a2.plot(epochs, count["n_occultable_total"], lw=1.4, color="navy",
            label="con parallasse")
    a2.plot(epochs, count["n_occultable_central"], lw=1.4, color="teal",
            ls="--", label="fascia centrale")
    a2.set_xlabel("Epoca [kyr dall'anno 0]")
    a2.set_ylabel(f"stelle V<{args.vmax} occultabili")
    a2.legend(fontsize=8)
    a2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "band.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    log("")
    log(f"Scritti {outdir}/spells.csv, count.csv, spells.png, band.png")


if __name__ == "__main__":
    main()
