#!/usr/bin/env python3
"""
constellation_visibility.py

How many stars of a sky culture were visible, and how many were not, as a
function of observer latitude and epoch.

A star of declination d culminates at altitude h = 90 - |phi - d| for an observer
at latitude phi, and rises at all only if h > 0. Counting, over the culture's
named stars, how many clear the horizon at each latitude and epoch is the whole
method. There is no fitted parameter anywhere in it.

Earlier versions of this file tried to extract a point estimate and a
significance test from the same data, through a step model, then a logistic
regression on altitude, then one on extinction. Each broke differently: the
threshold settled on the lowest canon star, the altitude regression followed the
canon's centroid in declination instead of the observer, the extinction one ran
its coefficient to infinity on a handful of stars near the horizon. The content
of the data is a constraint, phi <= 90 + d_min, not a point estimate, and the
honest output is the region satisfying it rather than a maximum inside it.

So the map is the result. Its plateau -- the cells where every named star rises
-- is bounded above by the southernmost star of the canon and below by the
northernmost, and it shifts with epoch because precession moves both. Where the
culture's known latitude enters and leaves that plateau is the epoch constraint.

Output
------
    summary.csv            one row per culture: the plateau's extent, where the
                           known latitude falls, and at which epochs

    maps/<culture>.csv     the map itself, one row per (epoch, latitude), with
                           the counts and the lowest culmination altitude

    maps/<culture>.pdf     the map drawn, contoured at 0, 1, 2, 5 and 10 stars
                           lost, with the known latitude and the present marked

Usage
-----
    python3 constellation_visibility.py \
        --trajectories BSGRID/trajectories.csv \
        --skycultures skycultures_csv \
        --outdir cv
"""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd

from star_table import read_star_table, normalise_epoch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Approximate observing latitude of each sky culture, in degrees, positive north.
# Coarse centroids of the region a culture is attached to, meant to be reviewed
# and overridden with --latitudes. For a seafaring culture a single latitude is a
# poor model of a voyaging range to begin with.
DEFAULT_LATITUDES: dict[str, float] = {
    "almagest": 31.2, "anutan": -11.6, "arabic": 24.0,
    "arabic_al-sufi": 32.6, "arabic_arabian_peninsula": 24.0,
    "arabic_indigenous": 24.0, "arabic_lunar_stations": 24.0,
    "aztec": 19.4, "babylonian_mulapin": 32.5, "babylonian_seleucid": 32.5,
    "belarusian": 53.9, "boorong": -35.5, "chinese": 34.3,
    "chinese_chenzhuo": 34.3, "chinese_contemporary": 34.3,
    "chinese_medieval": 34.3, "chinese_song_dynasty": 34.3,
    "dakota": 44.5, "egyptian": 25.7, "egyptian_dendera": 26.1,
    "greek_almagest": 31.2, "greek_dante": 43.8, "greek_farnese": 31.2,
    "greek_leidenAratea": 31.2, "hawaiian_starlines": 20.8,
    "indian": 25.0, "indian_nakshatras": 25.0, "inuit": 68.0,
    "japanese_moon_stations": 35.0, "kamilaroi": -30.0, "korean": 37.5,
    "lokono": 5.9, "macedonian": 41.6, "maori": -41.0, "maya": 20.7,
    "mongolian": 47.9, "navajo": 36.1, "norse": 60.0, "norse_edda": 64.0,
    "northern_andes": 4.6, "ojibwe": 47.0, "romanian": 45.9,
    "ruanui_sky_tahiti_and_society_islands": -17.6, "russian_siberian": 58.0,
    "sami": 68.4, "samoan": -13.8, "sardinian": 40.1, "seri": 29.0,
    "siberian": 58.0, "tibetan": 29.7, "tikuna": -4.0, "tongan": -21.1,
    "tukano": -0.5, "tupi": -10.0, "vanuatu_netwar": -19.5,
    "western": 37.0, "western_hlad": 37.0, "western_rey": 37.0,
}

# Sets covering the whole sky by construction: their plateau should be narrow
# and centred near the equator, since a canon spanning both poles can only be
# fully visible from near it. They are the sanity check on the whole map.
ALLSKY_CONTROLS = ("modern", "modern_chinese", "modern_hlad", "modern_iau",
                   "modern_journey_to_the_west", "modern_rey", "modern_st")

CONTOURS = (0, 1, 2, 5, 10)


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


def visibility_map(dec: np.ndarray, lat_grid: np.ndarray):
    """Counts and lowest culmination altitude over (epoch, latitude).

    dec is (n_epoch, n_star), declinations of date. Latitudes are looped over so
    the working array stays (n_epoch, n_star) rather than gaining a third axis.
    """
    n_epoch = dec.shape[0]
    n_lat = lat_grid.size

    n_visible = np.empty((n_epoch, n_lat), dtype=np.int32)
    h_low = np.empty((n_epoch, n_lat), dtype=float)

    for j, phi in enumerate(lat_grid):
        h = 90.0 - np.abs(phi - dec)
        n_visible[:, j] = (h > 0.0).sum(axis=1)
        h_low[:, j] = h.min(axis=1)

    return n_visible, h_low


def plateau_extent(n_invisible: np.ndarray, epochs, lat_grid, level: int):
    """Bounding box of the cells losing at most `level` stars."""
    ok = n_invisible <= level
    if not ok.any():
        return None
    return {
        "lat_lo": float(lat_grid[ok.any(axis=0)].min()),
        "lat_hi": float(lat_grid[ok.any(axis=0)].max()),
        "epoch_lo": float(epochs[ok.any(axis=1)].min()),
        "epoch_hi": float(epochs[ok.any(axis=1)].max()),
        "n_cells": int(ok.sum()),
        "frac_cells": float(ok.mean()),
    }


def plot_map(culture, epochs, lat_grid, n_invisible, lat_known,
             present_epoch, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    mesh = ax.pcolormesh(epochs, lat_grid, np.clip(n_invisible, 0, 20).T,
                         cmap="inferno_r", shading="auto", vmin=0, vmax=20)
    fig.colorbar(mesh, ax=ax, label="stelle del canone mai sopra l'orizzonte")

    levels = [l for l in CONTOURS if (n_invisible <= l).any()
              and (n_invisible > l).any()]
    if levels:
        cs = ax.contour(epochs, lat_grid, n_invisible.T,
                        levels=[l + 0.5 for l in levels],
                        colors="cyan", linewidths=1.2)
        ax.clabel(cs, fmt={l + 0.5: f"{l}" for l in levels}, fontsize=7)

    if lat_known is not None:
        ax.axhline(lat_known, color="lime", lw=1.5, ls="--",
                   label=f"latitudine nota {lat_known:+.1f}°")
    ax.axvline(present_epoch, color="white", lw=1.0, ls=":", label="presente")

    ax.set_xlabel("Epoca [kyr dall'anno 0]")
    ax.set_ylabel("Latitudine dell'osservatore [°]")
    ax.set_title(f"{culture} — stelle del canone perse sotto l'orizzonte")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Visible and invisible canon stars over latitude and epoch.")
    p.add_argument("--trajectories", required=True,
                   help="trajectories.csv from bright_star_grid.py")
    p.add_argument("--skycultures", required=True,
                   help="directory of CSV tables from stellarium_skycultures.py")
    p.add_argument("--outdir", default="cv")
    p.add_argument("--culture", default=None, help="restrict to one culture")
    p.add_argument("--latitudes", default=None,
                   help="CSV with columns culture,latitude overriding the built-in table")
    p.add_argument("--Tmin", type=float, default=-30.0, help="earliest epoch, kyr")
    p.add_argument("--Tmax", type=float, default=2.0, help="latest epoch, kyr")
    p.add_argument("--epoch-step", type=float, default=0.1, help="epoch step, kyr")
    p.add_argument("--lat-step", type=float, default=0.5, help="latitude step, deg")
    p.add_argument("--lat-min", type=float, default=-55.0)
    p.add_argument("--lat-max", type=float, default=70.0)
    p.add_argument("--present-kyr", type=float, default=2.0,
                   help="epoch taken as the present, kyr from year 0")
    p.add_argument("--min-stars", type=int, default=10,
                   help="skip cultures with fewer catalogued stars")
    args = p.parse_args()

    outdir = Path(args.outdir)
    (outdir / "maps").mkdir(parents=True, exist_ok=True)

    latitudes = dict(DEFAULT_LATITUDES)
    if args.latitudes:
        ext = pd.read_csv(args.latitudes)
        latitudes.update(dict(zip(ext["culture"], ext["latitude"].astype(float))))
        log(f"Latitude table overridden for {len(ext)} cultures")

    log(f"Reading {args.trajectories} ...")
    traj = read_star_table(args.trajectories,
                           ["epoch_kyr_from_year0", "HIP", "dec_deg"])
    traj = normalise_epoch(traj)
    log(f"  {len(traj):,} rows, {traj['HIP'].nunique()} stars, "
        f"{traj['epoch'].nunique()} epochs")

    available = np.sort(traj["epoch"].unique())
    keep = available[(available >= args.Tmin) & (available <= args.Tmax)]
    if keep.size == 0:
        raise SystemExit("No epoch in the trajectory file falls in the requested range.")
    base = np.min(np.diff(available)) if available.size > 1 else args.epoch_step
    stride = max(1, int(round(args.epoch_step / base)))
    traj = traj[traj["epoch"].isin(keep[::stride])]

    wide = traj.pivot(index="epoch", columns="HIP", values="dec_deg").sort_index()
    epochs = wide.index.to_numpy(dtype=float)
    all_hips = wide.columns.to_numpy()
    dec_all = wide.to_numpy(dtype=float)

    lat_grid = np.arange(args.lat_min, args.lat_max + 1e-9, args.lat_step)
    log(f"Epochs: {epochs.size} from {epochs.min():+.1f} to {epochs.max():+.1f} kyr")
    log(f"Latitudes: {lat_grid.size} from {lat_grid[0]:+.0f} to {lat_grid[-1]:+.0f}")

    present_epoch = float(epochs[np.argmin(np.abs(epochs - args.present_kyr))])
    i_present = int(np.argmin(np.abs(epochs - present_epoch)))
    log(f"Present taken as epoch {present_epoch:+.1f} kyr")

    members = pd.read_csv(Path(args.skycultures) / "members.csv")
    if args.culture:
        members = members[members["culture"] == args.culture]
        if members.empty:
            raise SystemExit(f"Culture not found: {args.culture}")

    cultures = sorted(members["culture"].unique())
    log(f"Cultures: {len(cultures)}")
    log("")

    rows = []
    for culture in cultures:
        hips = np.sort(members.loc[members["culture"] == culture, "HIP"].unique())
        mask = np.isin(all_hips, hips)
        n_canon = int(mask.sum())
        if n_canon < args.min_stars:
            log(f"  {culture:28s} SKIP: {n_canon} stelle nel catalogo")
            continue

        dec = dec_all[:, mask]
        n_visible, h_low = visibility_map(dec, lat_grid)
        n_invisible = n_canon - n_visible

        pd.DataFrame({
            "epoch_kyr": np.repeat(epochs, lat_grid.size),
            "lat_deg": np.tile(lat_grid, epochs.size),
            "n_visible": n_visible.ravel(),
            "n_invisible": n_invisible.ravel(),
            "frac_visible": n_visible.ravel() / n_canon,
            "h_low_deg": h_low.ravel(),
        }).to_csv(outdir / "maps" / f"{culture}.csv", index=False,
                  float_format="%.3f")

        lat_known = latitudes.get(culture)
        plot_map(culture, epochs, lat_grid, n_invisible, lat_known,
                 present_epoch, outdir / "maps" / f"{culture}.pdf")

        best = int(n_invisible.min())
        row = {
            "culture": culture,
            "is_allsky_control": culture in ALLSKY_CONTROLS,
            "n_stars_culture": len(hips),
            "n_stars_present": n_canon,
            "coverage": n_canon / len(hips),
            "min_invisible": best,
            "lat_known_deg": lat_known if lat_known is not None else np.nan,
        }

        # The plateau at each integer level: the region losing at most that many
        # stars. Level 0 is the hard constraint; the others show how fast the
        # region widens once a few stars are allowed to be missing.
        for level in CONTOURS:
            ext = plateau_extent(n_invisible, epochs, lat_grid, level)
            pre = f"lev{level}"
            if ext is None:
                row.update({f"{pre}_lat_lo": np.nan, f"{pre}_lat_hi": np.nan,
                            f"{pre}_epoch_lo": np.nan, f"{pre}_epoch_hi": np.nan,
                            f"{pre}_frac_cells": 0.0})
            else:
                row.update({f"{pre}_lat_lo": ext["lat_lo"],
                            f"{pre}_lat_hi": ext["lat_hi"],
                            f"{pre}_epoch_lo": ext["epoch_lo"],
                            f"{pre}_epoch_hi": ext["epoch_hi"],
                            f"{pre}_frac_cells": ext["frac_cells"]})

        if lat_known is None:
            row.update(n_invisible_at_known_now=np.nan,
                       epochs_ok_at_known=np.nan,
                       epoch_ok_lo=np.nan, epoch_ok_hi=np.nan,
                       present_ok=np.nan, notes="latitudine non nota")
        else:
            jk = int(np.argmin(np.abs(lat_grid - lat_known)))
            col = n_invisible[:, jk]
            ok = col == col.min()
            row.update(
                n_invisible_at_known_now=int(col[i_present]),
                min_invisible_at_known=int(col.min()),
                epochs_ok_at_known=int(ok.sum()),
                epoch_ok_lo=float(epochs[ok].min()),
                epoch_ok_hi=float(epochs[ok].max()),
                present_ok=bool(ok[i_present]),
                notes="",
            )
        rows.append(row)

        lat_txt = (f"lat nota {lat_known:+5.1f}°" if lat_known is not None
                   else "lat ignota    ")
        l0 = plateau_extent(n_invisible, epochs, lat_grid, 0)
        plateau = (f"[{l0['lat_lo']:+6.1f},{l0['lat_hi']:+6.1f}]°"
                   if l0 else "  nessuna cella a 0  ")
        extra = ("" if lat_known is None
                 else f" | perse oggi alla lat nota: {row['n_invisible_at_known_now']}")
        log(f"  {culture:28s} {n_canon:4d} st | min perse {best:3d} | "
            f"plateau {plateau} | {lat_txt}{extra}")

    if not rows:
        raise SystemExit("No culture could be analysed.")

    out = pd.DataFrame(rows)
    out.to_csv(outdir / "summary.csv", index=False)

    log("")
    log(f"Written {outdir}/summary.csv  ({len(out)} cultures)")
    known = out[out["lat_known_deg"].notna()]
    if not known.empty:
        log(f"  con latitudine nota: {len(known)}")
        log(f"    perdono zero stelle alla latitudine nota, oggi: "
            f"{int((known['n_invisible_at_known_now'] == 0).sum())}")
        log(f"    ne perdono qualcuna: "
            f"{int((known['n_invisible_at_known_now'] > 0).sum())}")
    ctrl = out[out["is_allsky_control"]]
    if not ctrl.empty:
        log("")
        log("  Controllo, insiemi a cielo intero (plateau atteso stretto "
            "e vicino all'equatore):")
        for _, r in ctrl.iterrows():
            log(f"    {r['culture']:28s} min perse {int(r['min_invisible']):3d} | "
                f"plateau lev0 [{r['lev0_lat_lo']:+6.1f},{r['lev0_lat_hi']:+6.1f}]°")


if __name__ == "__main__":
    main()
