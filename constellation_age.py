#!/usr/bin/env python3
"""
constellation_age.py

Date the origin of a sky culture from the empty region around a celestial pole.

Method
------
Ovenden (1966) and Roy (1984) dated the Greek constellations by noting that they
leave a blank region around the south celestial pole: that blank is the part of
the sky permanently below the horizon at the latitude where the constellations
were devised. Precession moves the pole, so the position of the blank fixes the
epoch and its size fixes the latitude. They obtained roughly 2000 BC at 36 N.

This program applies the same reasoning systematically, to any sky culture
parsed by stellarium_skycultures.py, using the star positions of date produced
by bright_star_grid.py:

  for each candidate epoch T
      take the constellation stars at their coordinates of date at T, in which
      the pole is by construction at declination -90 (or +90)
      find the largest star-free spherical cap anywhere on the sky
      record how far that cap's centre lies from the pole

  the best-fit epoch minimises that offset, and the cap radius R at that epoch
  implies an observing latitude |phi| = 90 - R

Two facts make the reasoning work: over a few millennia the constellation
pattern is nearly rigid, so the blank stays fixed with respect to the stars,
while the pole sweeps a 23.4-degree-radius circle every 25.8 kyr. The epoch is
read off the encounter between the two.

Applicability
-------------
The method is only meaningful for a culture that covers its visible sky densely.
A culture with a handful of constellations leaves large blanks everywhere, and
the largest one carries no information. Three diagnostics are therefore computed
and reported for every culture, and a verdict is derived from them:

  catalogue coverage  fraction of the culture's stars present in the trajectory
                      file; a bright-star catalogue misses faint constellation
                      members and manufactures blanks that are not real

  blank significance  Monte-Carlo probability that as large a blank arises from
                      the same number of stars scattered uniformly at random

  blank uniqueness    ratio of the largest blank to the largest blank disjoint
                      from it; near 1 means several equivalent blanks and no
                      distinguished invisible zone

A fourth number, the chance that a blank in a random direction is swept by the
pole path within the observed offset over the scanned epoch range, guards
against reading an epoch out of a coincidence.

Usage
-----
    python3 constellation_age.py \
        --trajectories BSGRID/trajectories.csv \
        --skycultures skycultures_csv \
        --Tmin -15 --Tmax 2 --epoch-step 0.1 \
        --outdir constellation_age_out
"""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.spatial import cKDTree
except ImportError as exc:
    raise SystemExit("Missing dependency 'scipy'. Install with: pip install scipy") from exc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OBLIQUITY_DEG = 23.44      # mean obliquity, sets the radius of the pole's circle
PRECESSION_PERIOD_KYR = 25.8


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Spherical geometry
# ---------------------------------------------------------------------------
def radec_to_xyz(ra_deg, dec_deg) -> np.ndarray:
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    cd = np.cos(dec)
    return np.column_stack((cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)))


def fibonacci_sphere(n: int) -> np.ndarray:
    """n roughly equidistant directions on the unit sphere.

    Mean spacing is about sqrt(4*pi/n) radians, so n = 100000 samples the sky at
    roughly 0.6 degrees, finer than the precision the method can support.
    """
    i = np.arange(n, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.column_stack((
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ))


def chord_to_angle(chord: np.ndarray) -> np.ndarray:
    """Euclidean chord length on the unit sphere to angular separation, degrees."""
    return np.degrees(2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0)))


def largest_empty_cap(star_xyz: np.ndarray, grid: np.ndarray):
    """Centre and angular radius of the largest star-free cap, and of the
    largest cap disjoint from it.

    The centre of a largest empty cap is a point of the sphere maximally distant
    from every star, found here by evaluating the nearest-star distance over a
    fixed grid of directions. The second, disjoint cap is the best centre that
    lies farther than the first radius from the first centre; comparing the two
    says whether the sky has one distinguished blank or several equivalent ones.
    """
    tree = cKDTree(star_xyz)
    chord, _ = tree.query(grid, k=1)
    ang = chord_to_angle(chord)

    i1 = int(np.argmax(ang))
    c1, r1 = grid[i1], float(ang[i1])

    sep1 = chord_to_angle(np.linalg.norm(grid - c1, axis=1))
    outside = sep1 > r1
    if outside.any():
        j = np.argmax(np.where(outside, ang, -1.0))
        r2 = float(ang[j])
    else:
        r2 = 0.0

    return c1, r1, r2


def angle_to_pole(xyz: np.ndarray, south: bool) -> float:
    """Angular distance from a direction to the south (or north) pole, degrees.

    Coordinates are of date, so the pole is the z axis by construction.
    """
    z = float(xyz[2])
    return float(np.degrees(np.arccos(np.clip(-z if south else z, -1.0, 1.0))))


# ---------------------------------------------------------------------------
# Null models
# ---------------------------------------------------------------------------
def null_blank_radius(n_stars: int, grid: np.ndarray, n_trials: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Largest empty cap radius for n_stars uniform random directions."""
    out = np.empty(n_trials)
    for k in range(n_trials):
        v = rng.normal(size=(n_stars, 3))
        v /= np.linalg.norm(v, axis=1)[:, None]
        chord, _ = cKDTree(v).query(grid, k=1)
        out[k] = chord_to_angle(chord).max()
    return out


def pole_path_probability(offset_deg: float, epoch_span_kyr: float) -> float:
    """Chance that a blank in a random direction lies within offset of the pole
    path swept over the scanned epoch range.

    The pole traces a small circle of angular radius equal to the obliquity, of
    circumference 2*pi*sin(eps), completing it in one precession period. A band
    of half-width `offset` around the traversed arc covers a solid angle of
    about arc * 2 * offset, to be compared with the whole sphere.
    """
    frac = min(1.0, abs(epoch_span_kyr) / PRECESSION_PERIOD_KYR)
    arc = 2.0 * np.pi * np.sin(np.radians(OBLIQUITY_DEG)) * frac
    band = arc * 2.0 * np.radians(offset_deg)
    return float(min(1.0, band / (4.0 * np.pi)))


# ---------------------------------------------------------------------------
# Per-culture analysis
# ---------------------------------------------------------------------------
def analyse_culture(culture: str, hips: np.ndarray, traj: pd.DataFrame,
                    epochs: np.ndarray, grid: np.ndarray, south: bool,
                    null_radii: np.ndarray | None):
    """Scan the epoch range and return the fit curve plus a summary row."""
    sub = traj[traj["HIP"].isin(hips)]
    present = np.sort(sub["HIP"].unique())
    coverage = len(present) / len(hips) if len(hips) else 0.0

    rows = []
    for t in epochs:
        e = sub[sub["epoch"] == t]
        if len(e) < 4:
            continue
        xyz = radec_to_xyz(e["ra_deg"].to_numpy(), e["dec_deg"].to_numpy())
        c1, r1, r2 = largest_empty_cap(xyz, grid)
        rows.append({
            "epoch_kyr": t,
            "blank_radius_deg": r1,
            "blank_second_deg": r2,
            "offset_from_pole_deg": angle_to_pole(c1, south),
            "implied_latitude_deg": 90.0 - r1 if south else -(90.0 - r1),
        })

    curve = pd.DataFrame(rows)
    if curve.empty:
        return curve, None

    best = curve.loc[curve["offset_from_pole_deg"].idxmin()]

    uniqueness = (best["blank_radius_deg"] / best["blank_second_deg"]
                  if best["blank_second_deg"] > 0 else np.inf)
    p_blank = (float((null_radii >= best["blank_radius_deg"]).mean())
               if null_radii is not None else np.nan)
    p_epoch = pole_path_probability(best["offset_from_pole_deg"],
                                    epochs.max() - epochs.min())

    # A result is reported as usable only if the blank is real, singular, and
    # not explainable by the pole happening to sweep past an arbitrary gap.
    reasons = []
    if coverage < 0.5:
        reasons.append(f"copertura catalogo {coverage:.0%}")
    if not np.isnan(p_blank) and p_blank > 0.05:
        reasons.append(f"vuoto non significativo (p={p_blank:.2f})")
    if uniqueness < 1.3:
        reasons.append(f"vuoto non unico (R1/R2={uniqueness:.2f})")
    if p_epoch > 0.05:
        reasons.append(f"epoca non vincolata (p={p_epoch:.2f})")

    summary = {
        "culture": culture,
        "n_stars_culture": len(hips),
        "n_stars_present": len(present),
        "coverage": coverage,
        "best_epoch_kyr": best["epoch_kyr"],
        "best_epoch_year": best["epoch_kyr"] * 1000.0,
        "implied_latitude_deg": best["implied_latitude_deg"],
        "blank_radius_deg": best["blank_radius_deg"],
        "blank_second_deg": best["blank_second_deg"],
        "blank_uniqueness": uniqueness,
        "offset_from_pole_deg": best["offset_from_pole_deg"],
        "p_blank_random": p_blank,
        "p_epoch_coincidence": p_epoch,
        "usable": len(reasons) == 0,
        "notes": "; ".join(reasons),
    }
    return curve, summary


def plot_culture(culture: str, curve: pd.DataFrame, summary: dict, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(curve["epoch_kyr"], curve["offset_from_pole_deg"], lw=1.2)
    ax1.axvline(summary["best_epoch_kyr"], color="crimson", lw=1.0, ls="--")
    ax1.set_ylabel("Scarto centro del vuoto – polo [°]")
    ax1.set_title(
        f"{culture}: epoca migliore {summary['best_epoch_year']:+.0f} "
        f"(anno astronomico), latitudine implicata "
        f"{summary['implied_latitude_deg']:+.1f}°"
        + ("" if summary["usable"] else "   [NON UTILIZZABILE]")
    )
    ax1.grid(alpha=0.3)

    ax2.plot(curve["epoch_kyr"], curve["blank_radius_deg"], lw=1.2, label="vuoto maggiore")
    ax2.plot(curve["epoch_kyr"], curve["blank_second_deg"], lw=1.0, ls=":",
             label="secondo vuoto disgiunto")
    ax2.axvline(summary["best_epoch_kyr"], color="crimson", lw=1.0, ls="--")
    ax2.set_xlabel("Epoca [kyr dall'anno 0]")
    ax2.set_ylabel("Raggio [°]")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Date sky cultures from the empty zone around a celestial pole.")
    p.add_argument("--trajectories", required=True,
                   help="trajectories.csv from bright_star_grid.py")
    p.add_argument("--skycultures", required=True,
                   help="directory of CSV tables from stellarium_skycultures.py")
    p.add_argument("--outdir", default="constellation_age_out")
    p.add_argument("--culture", default=None, help="restrict to one culture")
    p.add_argument("--Tmin", type=float, default=-15.0, help="earliest epoch, kyr")
    p.add_argument("--Tmax", type=float, default=2.0, help="latest epoch, kyr")
    p.add_argument("--epoch-step", type=float, default=0.1, help="epoch step, kyr")
    p.add_argument("--grid", type=int, default=100_000,
                   help="directions used to locate the empty cap")
    p.add_argument("--min-stars", type=int, default=20,
                   help="skip cultures with fewer catalogued stars")
    p.add_argument("--null-trials", type=int, default=200,
                   help="Monte-Carlo trials for the blank-significance test")
    p.add_argument("--north", action="store_true",
                   help="look for the blank at the north pole (southern cultures)")
    p.add_argument("--seed", type=int, default=20260814)
    args = p.parse_args()

    outdir = Path(args.outdir)
    (outdir / "curves").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    log(f"Reading {args.trajectories} ...")
    traj = pd.read_csv(args.trajectories,
                       usecols=["epoch_kyr_from_year0", "HIP", "ra_deg", "dec_deg"])
    traj = traj.rename(columns={"epoch_kyr_from_year0": "epoch"})
    traj["epoch"] = traj["epoch"].round(6)
    log(f"  {len(traj):,} rows, {traj['HIP'].nunique()} stars, "
        f"{traj['epoch'].nunique()} epochs")

    available = np.sort(traj["epoch"].unique())
    wanted = available[(available >= args.Tmin) & (available <= args.Tmax)]
    if wanted.size == 0:
        raise SystemExit("No epoch in the trajectory file falls in the requested range.")
    stride = max(1, int(round(args.epoch_step / np.min(np.diff(available)))) ) if available.size > 1 else 1
    epochs = wanted[::stride]
    log(f"Scanning {epochs.size} epochs, {epochs.min():+.1f} .. {epochs.max():+.1f} kyr")

    members = pd.read_csv(Path(args.skycultures) / "members.csv")
    if args.culture:
        members = members[members["culture"] == args.culture]
        if members.empty:
            raise SystemExit(f"Culture not found: {args.culture}")

    grid = fibonacci_sphere(args.grid)
    south = not args.north
    log(f"Blank sought at the {'south' if south else 'north'} celestial pole "
        f"on {args.grid:,} directions")

    cultures = sorted(members["culture"].unique())
    log(f"Cultures to analyse: {len(cultures)}")

    null_cache: dict[int, np.ndarray] = {}
    summaries = []

    for culture in cultures:
        hips = np.sort(members.loc[members["culture"] == culture, "HIP"].unique())
        n_present = traj.loc[traj["HIP"].isin(hips), "HIP"].nunique()
        if n_present < args.min_stars:
            log(f"  {culture:28s} SKIP: {n_present} stelle nel catalogo "
                f"(minimo {args.min_stars})")
            continue

        # The null depends only on how many stars are on the sky, so it is
        # computed once per distinct count and reused.
        if n_present not in null_cache:
            null_cache[n_present] = null_blank_radius(
                n_present, grid, args.null_trials, rng)

        curve, summary = analyse_culture(culture, hips, traj, epochs, grid,
                                         south, null_cache[n_present])
        if summary is None:
            log(f"  {culture:28s} SKIP: nessuna epoca utilizzabile")
            continue

        curve.to_csv(outdir / "curves" / f"{culture}.csv", index=False)
        plot_culture(culture, curve, summary, outdir / "curves" / f"{culture}.pdf")
        summaries.append(summary)

        flag = "OK " if summary["usable"] else "-- "
        log(f"  {flag}{culture:26s} epoca {summary['best_epoch_year']:+7.0f} | "
            f"lat {summary['implied_latitude_deg']:+5.1f}° | "
            f"vuoto {summary['blank_radius_deg']:4.1f}° | "
            f"scarto {summary['offset_from_pole_deg']:4.1f}° | "
            f"cop. {summary['coverage']:.0%}"
            + (f" | {summary['notes']}" if summary["notes"] else ""))

    if not summaries:
        raise SystemExit("No culture could be analysed.")

    out = pd.DataFrame(summaries).sort_values(
        ["usable", "offset_from_pole_deg"], ascending=[False, True])
    out.to_csv(outdir / "summary.csv", index=False)

    log("")
    log(f"Written {outdir}/summary.csv  ({len(out)} cultures, "
        f"{int(out['usable'].sum())} usable)")


if __name__ == "__main__":
    main()
