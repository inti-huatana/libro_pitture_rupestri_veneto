#!/usr/bin/env python3
"""
constellation_visibility.py

Constrain the observing latitude and the epoch of a sky culture from the
requirement that every star it names was actually visible.

Supersedes the empty-zone approach of constellation_age.py, which inferred from
an absence (no constellations near the south celestial pole) and therefore only
worked for northern cultures. This one infers from a presence: a star belongs to
the culture's canon, so somebody saw it, so it rose above the horizon. That is a
physical constraint rather than a fit, and it is symmetric in the hemispheres,
so it applies unchanged to Maori, Inuit or Babylonian material.

Method
------
A star of declination d culminates at altitude 90 - |phi - d| for an observer at
latitude phi. Requiring it to reach at least h_min gives |phi - d| <= 90 - h_min.
Imposing that on the whole canon at epoch T bounds the latitude:

    d_max(T) - 90 + h_min  <=  phi  <=  d_min(T) + 90 - h_min

so the admissible band is 180 - (declination span) - 2*h_min degrees wide: the
constraint bites in proportion to how far in declination the canon reaches.

h_min is the practical horizon. A star culminating two degrees up is not usable
for building a figure -- some two magnitudes of atmospheric extinction, plus haze
and relief -- so h_min is an explicit parameter here rather than a hidden bias,
and every result is reported for several of its values.

Because the extremes d_min and d_max are order statistics, one misattributed
star in a modern reconstruction would destroy the bound. The band is therefore
also computed while tolerating k stars below the practical horizon, for several
k: a result that moves a lot between k=0 and k=3 is driven by an outlier and
should not be believed.

Precession sweeps declinations by up to +/-23.4 degrees, so the band edges move
substantially with epoch. Given a culture's latitude, which ethnography usually
supplies, the latitude constraint inverts into a constraint on the epochs at
which the canon could have been assembled. Its period is that of precession,
25.8 kyr, so within the Holocene the solution is unique.

What to look for
----------------
Modern reconstructions are built by scholars who know where the culture lived
and would not have included stars invisible from there today, so finding a canon
visible from its own latitude at the present epoch is close to circular. The
informative case is the opposite one, flagged as `informative` in the summary:

    the present epoch is NOT admissible, while some past epoch is

that cannot be an artefact of the reconstruction, and marks a canon that only
makes sense as an inheritance from an earlier sky.

Expect few such cultures. For most the constraint will be wide and the present
admissible, which is the correct answer rather than a failure of the method.

Usage
-----
    python3 constellation_visibility.py \
        --trajectories BSGRID/trajectories.csv \
        --skycultures skycultures_csv \
        --outdir constellation_visibility_out
"""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Approximate observing latitude of each sky culture, in degrees, positive north.
# These are coarse centroids of the region a culture is attached to, meant to be
# reviewed and overridden with --latitudes; they carry no better precision than a
# degree or two, and for seafaring cultures a single latitude is a poor model of
# a voyaging range in the first place. A culture absent from this table is still
# analysed, but only its two-dimensional admissible region is reported, since
# without a latitude there is no epoch constraint to extract.
DEFAULT_LATITUDES: dict[str, float] = {
    "almagest": 31.2,                    # Alexandria
    "anutan": -11.6,                     # Anuta, Solomon Is.
    "arabic": 24.0,
    "arabic_al-sufi": 32.6,              # Isfahan
    "arabic_arabian_peninsula": 24.0,
    "arabic_indigenous": 24.0,
    "arabic_lunar_stations": 24.0,
    "aztec": 19.4,                       # Tenochtitlan
    "babylonian_mulapin": 32.5,          # Babylon
    "babylonian_seleucid": 32.5,
    "belarusian": 53.9,
    "boorong": -35.5,                    # NW Victoria, Australia
    "chinese": 34.3,                     # Xi'an
    "chinese_contemporary": 34.3,
    "chinese_medieval": 34.3,
    "dakota": 44.5,
    "egyptian": 25.7,                    # Thebes
    "hawaiian_starlines": 20.8,
    "indian": 25.0,
    "inuit": 68.0,
    "japanese_moon_stations": 35.0,
    "kamilaroi": -30.0,                  # NE New South Wales
    "korean": 37.5,
    "lokono": 5.9,                       # Guianas
    "macedonian": 41.6,
    "maori": -41.0,                      # Aotearoa / New Zealand
    "maya": 20.7,
    "mongolian": 47.9,
    "navajo": 36.1,
    "norse": 60.0,
    "northern_andes": 4.6,
    "ojibwe": 47.0,
    "romanian": 45.9,
    "russian_siberian": 58.0,
    "sami": 68.4,
    "sardinian": 40.1,
    "seri": 29.0,                        # Sonora, Mexico
    "siberian": 58.0,
    "tongan": -21.1,
    "tukano": -0.5,                      # NW Amazon
    "tupi": -10.0,
    "vanuatu": -16.0,
    "western": 37.0,                     # Greek tradition, Mediterranean
    "western_hlad": 37.0,
    "western_rey": 37.0,
}


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def violation_map(dec: np.ndarray, lat_grid: np.ndarray, h_min: float) -> np.ndarray:
    """Number of canon stars that never reach h_min, per (epoch, latitude).

    dec has shape (n_epoch, n_star) and holds declinations of date. A star is
    counted when its culmination altitude 90 - |phi - d| falls below h_min, i.e.
    when it lies outside the declination window the latitude can ever lift that
    high. Latitudes are looped over rather than broadcast so that the temporary
    stays of size (n_epoch, n_star) instead of (n_epoch, n_star, n_lat).
    """
    out = np.empty((dec.shape[0], lat_grid.size), dtype=np.int32)
    reach = 90.0 - h_min
    for j, phi in enumerate(lat_grid):
        out[:, j] = ((dec < phi - reach) | (dec > phi + reach)).sum(axis=1)
    return out


def band_edges(viol: np.ndarray, lat_grid: np.ndarray, k: int):
    """Lowest and highest latitude tolerating at most k unseen stars, per epoch.

    Also reports whether the admissible set is a single interval. It usually is,
    but nothing guarantees it: the count rises with latitude as southern stars
    drop out and falls as northern ones come in, and a canon clumped at both
    declination extremes can in principle split the set in two, which would make
    a plain min/max summary misleading.
    """
    ok = viol <= k
    n_epoch = viol.shape[0]
    lo = np.full(n_epoch, np.nan)
    hi = np.full(n_epoch, np.nan)
    contiguous = np.ones(n_epoch, dtype=bool)

    for i in range(n_epoch):
        idx = np.flatnonzero(ok[i])
        if idx.size == 0:
            contiguous[i] = True          # empty set, nothing to be split
            continue
        lo[i] = lat_grid[idx[0]]
        hi[i] = lat_grid[idx[-1]]
        contiguous[i] = idx.size == (idx[-1] - idx[0] + 1)

    return lo, hi, contiguous


def admissible_epochs(viol: np.ndarray, lat_grid: np.ndarray,
                      epochs: np.ndarray, lat_known: float, k: int):
    """Epochs at which the canon is visible from a known latitude."""
    j = int(np.argmin(np.abs(lat_grid - lat_known)))
    ok = viol[:, j] <= k
    return epochs[ok], ok


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_culture(culture: str, epochs: np.ndarray, lat_grid: np.ndarray,
                 viol: np.ndarray, lat_known: float | None,
                 present_epoch: float, h_min: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))

    # Shade by how many stars fail, saturating early: the interesting contrast is
    # between none, a few and many, not the exact count deep in the excluded zone.
    shown = np.clip(viol, 0, 10)
    mesh = ax.pcolormesh(epochs, lat_grid, shown.T, cmap="inferno_r",
                         shading="auto", vmin=0, vmax=10)
    fig.colorbar(mesh, ax=ax, label="stelle mai sopra l'orizzonte pratico")

    ax.contour(epochs, lat_grid, viol.T, levels=[0.5], colors="cyan", linewidths=1.6)

    if lat_known is not None:
        ax.axhline(lat_known, color="lime", lw=1.4, ls="--",
                   label=f"latitudine nota {lat_known:+.1f}°")
    ax.axvline(present_epoch, color="white", lw=1.0, ls=":", label="presente")

    ax.set_xlabel("Epoca [kyr dall'anno 0]")
    ax.set_ylabel("Latitudine dell'osservatore [°]")
    ax.set_title(f"{culture} — regione ammissibile (h_min = {h_min:.0f}°); "
                 f"il contorno ciano racchiude zero violazioni")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Latitude and epoch constraints from canon visibility.")
    p.add_argument("--trajectories", required=True,
                   help="trajectories.csv from bright_star_grid.py")
    p.add_argument("--skycultures", required=True,
                   help="directory of CSV tables from stellarium_skycultures.py")
    p.add_argument("--outdir", default="constellation_visibility_out")
    p.add_argument("--culture", default=None, help="restrict to one culture")
    p.add_argument("--latitudes", default=None,
                   help="CSV with columns culture,latitude overriding the built-in table")
    p.add_argument("--Tmin", type=float, default=-30.0, help="earliest epoch, kyr")
    p.add_argument("--Tmax", type=float, default=2.0, help="latest epoch, kyr")
    p.add_argument("--epoch-step", type=float, default=0.1, help="epoch step, kyr")
    p.add_argument("--lat-step", type=float, default=0.5, help="latitude step, deg")
    p.add_argument("--h-min", type=float, nargs="+", default=[0.0, 5.0, 10.0],
                   help="practical horizon altitudes to report, deg")
    p.add_argument("--k-outliers", type=int, nargs="+", default=[0, 1, 3],
                   help="tolerated unseen stars")
    p.add_argument("--present-kyr", type=float, default=2.0,
                   help="epoch taken as the present, kyr from year 0")
    p.add_argument("--min-stars", type=int, default=10,
                   help="skip cultures with fewer catalogued stars")
    p.add_argument("--ref-h-min", type=float, default=5.0,
                   help="h_min used for the per-culture plot")
    args = p.parse_args()

    outdir = Path(args.outdir)
    (outdir / "bands").mkdir(parents=True, exist_ok=True)
    (outdir / "maps").mkdir(parents=True, exist_ok=True)

    latitudes = dict(DEFAULT_LATITUDES)
    if args.latitudes:
        ext = pd.read_csv(args.latitudes)
        latitudes.update(dict(zip(ext["culture"], ext["latitude"].astype(float))))
        log(f"Latitude table overridden for {len(ext)} cultures")

    log(f"Reading {args.trajectories} ...")
    traj = pd.read_csv(args.trajectories,
                       usecols=["epoch_kyr_from_year0", "HIP", "dec_deg"])
    traj = traj.rename(columns={"epoch_kyr_from_year0": "epoch"})
    traj["epoch"] = traj["epoch"].round(6)
    log(f"  {len(traj):,} rows, {traj['HIP'].nunique()} stars, "
        f"{traj['epoch'].nunique()} epochs")

    available = np.sort(traj["epoch"].unique())
    keep = available[(available >= args.Tmin) & (available <= args.Tmax)]
    if keep.size == 0:
        raise SystemExit("No epoch in the trajectory file falls in the requested range.")
    base_step = np.min(np.diff(available)) if available.size > 1 else args.epoch_step
    stride = max(1, int(round(args.epoch_step / base_step)))
    epochs = keep[::stride]
    traj = traj[traj["epoch"].isin(epochs)]
    log(f"Epochs: {epochs.size} from {epochs.min():+.1f} to {epochs.max():+.1f} kyr")

    lat_grid = np.arange(-90.0, 90.0 + 1e-9, args.lat_step)
    log(f"Latitudes: {lat_grid.size} from {lat_grid[0]:+.0f} to {lat_grid[-1]:+.0f}")

    present_epoch = float(epochs[np.argmin(np.abs(epochs - args.present_kyr))])
    log(f"Present taken as epoch {present_epoch:+.1f} kyr")

    members = pd.read_csv(Path(args.skycultures) / "members.csv")
    if args.culture:
        members = members[members["culture"] == args.culture]
        if members.empty:
            raise SystemExit(f"Culture not found: {args.culture}")

    cultures = sorted(members["culture"].unique())
    log(f"Cultures: {len(cultures)}")
    log("")

    summaries = []

    for culture in cultures:
        hips = np.sort(members.loc[members["culture"] == culture, "HIP"].unique())
        sub = traj[traj["HIP"].isin(hips)]
        n_present = sub["HIP"].nunique()
        if n_present < args.min_stars:
            log(f"  {culture:28s} SKIP: {n_present} stelle nel catalogo")
            continue

        mat = sub.pivot(index="epoch", columns="HIP", values="dec_deg").sort_index()
        dec = mat.to_numpy(dtype=float)
        ep = mat.index.to_numpy(dtype=float)
        lat_known = latitudes.get(culture)

        i_present = int(np.argmin(np.abs(ep - present_epoch)))
        dec_span_now = float(dec[i_present].max() - dec[i_present].min())

        band_rows = []
        for h_min in args.h_min:
            viol = violation_map(dec, lat_grid, h_min)

            if abs(h_min - args.ref_h_min) < 1e-9:
                np.save(outdir / "maps" / f"{culture}_hmin{h_min:.0f}.npy", viol)
                plot_culture(culture, ep, lat_grid, viol, lat_known,
                             present_epoch, h_min,
                             outdir / "maps" / f"{culture}.pdf")

            for k in args.k_outliers:
                lo, hi, contig = band_edges(viol, lat_grid, k)
                for i in range(ep.size):
                    band_rows.append({
                        "culture": culture, "epoch_kyr": ep[i],
                        "h_min_deg": h_min, "k_outliers": k,
                        "lat_lo_deg": lo[i], "lat_hi_deg": hi[i],
                        "contiguous": contig[i],
                        "dec_min_deg": dec[i].min(), "dec_max_deg": dec[i].max(),
                    })

                row = {
                    "culture": culture,
                    "n_stars_culture": len(hips),
                    "n_stars_present": n_present,
                    "coverage": n_present / len(hips),
                    "h_min_deg": h_min, "k_outliers": k,
                    "dec_span_present_deg": dec_span_now,
                    "band_width_present_deg": hi[i_present] - lo[i_present],
                    "lat_lo_present_deg": lo[i_present],
                    "lat_hi_present_deg": hi[i_present],
                    "lat_known_deg": lat_known if lat_known is not None else np.nan,
                }

                if lat_known is None:
                    row.update(present_admissible=np.nan, n_epochs_admissible=np.nan,
                               frac_epochs_admissible=np.nan,
                               epoch_admissible_min=np.nan, epoch_admissible_max=np.nan,
                               informative=False,
                               notes="latitudine non nota: solo regione 2D")
                else:
                    good_ep, ok = admissible_epochs(viol, lat_grid, ep, lat_known, k)
                    present_ok = bool(ok[i_present])
                    row.update(
                        present_admissible=present_ok,
                        n_epochs_admissible=int(ok.sum()),
                        frac_epochs_admissible=float(ok.mean()),
                        epoch_admissible_min=float(good_ep.min()) if good_ep.size else np.nan,
                        epoch_admissible_max=float(good_ep.max()) if good_ep.size else np.nan,
                        # The one case a modern reconstruction cannot have
                        # manufactured: unusable today, usable in the past.
                        informative=bool((not present_ok) and ok.any()),
                        notes="",
                    )
                summaries.append(row)

        pd.DataFrame(band_rows).to_csv(outdir / "bands" / f"{culture}.csv", index=False)

        ref = [s for s in summaries
               if s["culture"] == culture
               and abs(s["h_min_deg"] - args.ref_h_min) < 1e-9
               and s["k_outliers"] == 0]
        if ref:
            r = ref[0]
            lat_txt = (f"lat nota {r['lat_known_deg']:+5.1f}°"
                       if not np.isnan(r["lat_known_deg"]) else "lat ignota    ")
            if np.isnan(r["lat_known_deg"]):
                verdict = "--"
            elif r["informative"]:
                verdict = "INFORMATIVA"
            elif r["present_admissible"]:
                verdict = "presente ammesso"
            else:
                verdict = "nessuna epoca ammessa"
            log(f"  {culture:28s} {n_present:4d} st | span {dec_span_now:5.1f}° | "
                f"banda oggi [{r['lat_lo_present_deg']:+6.1f},{r['lat_hi_present_deg']:+6.1f}] | "
                f"{lat_txt} | {verdict}")

    if not summaries:
        raise SystemExit("No culture could be analysed.")

    out = pd.DataFrame(summaries)
    out.to_csv(outdir / "summary.csv", index=False)

    ref = out[(np.abs(out["h_min_deg"] - args.ref_h_min) < 1e-9)
              & (out["k_outliers"] == 0)]
    log("")
    log(f"Written {outdir}/summary.csv  ({len(out)} rows, "
        f"{out['culture'].nunique()} cultures)")
    log(f"  reference h_min={args.ref_h_min:.0f}°, k=0:")
    log(f"    culture informative (presente escluso, passato ammesso): "
        f"{int(ref['informative'].sum())}")
    log(f"    presente ammesso                                       : "
        f"{int((ref['present_admissible'] == True).sum())}")
    log(f"    nessuna epoca ammessa                                  : "
        f"{int(((ref['present_admissible'] == False) & (~ref['informative'])).sum())}")
    log(f"    latitudine non nota                                    : "
        f"{int(ref['lat_known_deg'].isna().sum())}")


if __name__ == "__main__":
    main()
