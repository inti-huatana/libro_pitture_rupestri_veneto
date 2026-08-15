#!/usr/bin/env python3
"""
horizon_geometry.py

Where the Sun, the Moon, the Milky Way and the stars meet the horizon, at a
given latitude, through time.

All of it is one question asked of different bodies: a body of declination d
rises, at latitude phi, at the azimuth given by cos A = sin d / cos phi. The Sun
at the solstices has d equal to the obliquity, the Moon at its standstills the
obliquity plus or minus its orbital inclination, and a star simply its own
declination. Nothing here needs an ephemeris, because none of it asks where a
body was on a given night, only how far along the horizon it could ever get.

Three things follow.

    marker stars    which stars rise where the solstice Sun or the standstill
                    Moon stops. Such a star holds the point when the body is
                    absent, and the pairing is checkable rather than supposed,
                    both declinations being computed. Since the obliquity and
                    the stars both move, the pairings form and dissolve, and
                    when they do is the output.

    solstice drift  how far the solstice rising point wanders over the span.
                    At the latitude of the Veneto it is close to four degrees,
                    more than the tolerance any alignment is claimed to. A site
                    said to face midsummer faces it at one epoch and not at
                    another, and the figure says which.

    Milky Way       the angle its plane makes with the horizon. The band as
                    river, road or path is the most widespread of all sky
                    motifs, and whether it stands upright or lies flat depends
                    on latitude, on the hour, and on precession.

Requires earth_orientation.py to have been run: the obliquity and the galactic
pole of date come from its table.

Usage
-----
    python3 horizon_geometry.py --trajectories BSGRID/trajectories.csv \
        --orientation earth/orientation.csv --lat 45.5 --outdir horizon
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd

from star_table import (normalise_epoch, pivot_epoch_star,
                        read_star_table, star_labels)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# How close a star's declination must come to a body's for the two to share a
# rising point. A quarter of a degree is half the Sun's disc, and below the
# precision with which anyone sights a notch on a ridge.
MATCH_TOL_DEG = 0.25


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


def rise_azimuth(dec_deg, lat_deg):
    """Azimuth from north of the rising point. NaN when the body never rises
    or never sets, where a rising point does not exist."""
    d = np.radians(np.asarray(dec_deg, dtype=float))
    c = np.sin(d) / math.cos(math.radians(lat_deg))
    c = np.where(np.abs(c) <= 1.0, c, np.nan)
    return np.degrees(np.arccos(c))


def milky_way_tilt(gp_ra_deg, gp_dec_deg, lat_deg, lst_deg):
    """Angle between the galactic plane and the horizon, degrees.

    The galactic plane meets the horizon in a line, and the angle between the
    two planes is the complement of the angle between their poles. The horizon's
    pole is the zenith, so the tilt is ninety degrees minus the zenith distance
    of the galactic pole -- that is, its altitude, taken without sign: a
    galactic pole at the zenith lays the band flat around the horizon, a pole on
    the horizon stands the band upright through the zenith.
    """
    ra = np.radians(gp_ra_deg)
    dec = np.radians(gp_dec_deg)
    phi = math.radians(lat_deg)
    h = np.radians(lst_deg) - ra                        # hour angle
    sin_alt = (np.sin(dec) * math.sin(phi)
               + np.cos(dec) * math.cos(phi) * np.cos(h))
    alt = np.degrees(np.arcsin(np.clip(sin_alt, -1.0, 1.0)))
    return np.abs(alt)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Horizon points of Sun, Moon, Milky Way and stars.")
    p.add_argument("--trajectories", required=True)
    p.add_argument("--orientation", required=True,
                   help="orientation.csv from earth_orientation.py")
    p.add_argument("--lat", type=float, default=45.5,
                   help="observer latitude, degrees")
    p.add_argument("--outdir", default="horizon")
    p.add_argument("--vmax", type=float, default=3.0,
                   help="only stars this bright can hold a horizon point")
    p.add_argument("--tol", type=float, default=MATCH_TOL_DEG,
                   help="declination tolerance for a shared rising point, deg")
    p.add_argument("--min-span", type=float, default=0.3,
                   help="shortest pairing reported, kyr")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log(f"Leggo {args.orientation} ...")
    ori = pd.read_csv(args.orientation).sort_values("epoch_kyr")

    log(f"Leggo {args.trajectories} ...")
    traj = read_star_table(args.trajectories,
                           ["epoch_kyr_from_year0", "HIP", "dec_deg", "Vmag",
                            "NAME", "Bayer"])
    traj = normalise_epoch(traj).drop_duplicates(["HIP", "epoch"])

    labels = star_labels(traj)
    epochs, hips, _arr = pivot_epoch_star(traj, ["dec_deg", "Vmag"])
    dec, mag = _arr["dec_deg"], _arr["Vmag"]

    # Put the orientation table on the star epochs.
    def on_epochs(col):
        return np.interp(epochs, ori["epoch_kyr"].to_numpy(), ori[col].to_numpy())

    eps = on_epochs("obliquity_deg")
    gp_ra = on_epochs("gal_pole_ra_deg")
    gp_dec = on_epochs("gal_pole_dec_deg")
    log(f"  {hips.size} stelle, {epochs.size} epoche "
        f"[{epochs.min():+.1f}, {epochs.max():+.1f}] kyr | lat {args.lat:+.1f}°")

    bodies = {
        "sole_solstizio_estivo": eps,
        "sole_solstizio_invernale": -eps,
        "luna_lunistizio_maggiore_N": eps + 5.145,
        "luna_lunistizio_maggiore_S": -(eps + 5.145),
        "luna_lunistizio_minore_N": eps - 5.145,
        "luna_lunistizio_minore_S": -(eps - 5.145),
    }

    # ---- horizon points of the bodies -----------------------------------
    rows = []
    for name, d in bodies.items():
        az = rise_azimuth(d, args.lat)
        for i in range(epochs.size):
            rows.append({"epoch_kyr": epochs[i], "body": name,
                         "dec_deg": d[i], "rise_azimuth_deg": az[i]})
    bodies_df = pd.DataFrame(rows)
    bodies_df.to_csv(outdir / "bodies.csv", index=False, float_format="%.4f")

    log("")
    log("  Escursione dell'azimut di sorgere su tutto il periodo:")
    for name in bodies:
        a = bodies_df[bodies_df.body == name]["rise_azimuth_deg"].dropna()
        if a.empty:
            log(f"    {name:28s} mai sorge/tramonta a questa latitudine")
        else:
            log(f"    {name:28s} {a.min():6.2f}° .. {a.max():6.2f}°  "
                f"escursione {a.max()-a.min():5.2f}°")

    # ---- stars sharing those points --------------------------------------
    bright = mag <= args.vmax
    match_rows = []
    for name, d in bodies.items():
        near = bright & (np.abs(dec - d[:, None]) <= args.tol)
        for k in range(hips.size):
            idx = np.flatnonzero(near[:, k])
            if idx.size == 0:
                continue
            # contiguous stretches of epochs where the pairing holds
            splits = np.flatnonzero(np.diff(idx) > 1)
            for seg in np.split(idx, splits + 1):
                span = abs(epochs[seg[-1]] - epochs[seg[0]])
                if span < args.min_span:
                    continue
                j = seg[int(np.argmin(np.abs(dec[seg, k] - d[seg])))]
                match_rows.append({
                    "body": name, "HIP": int(hips[k]),
                    "star": labels.get(int(hips[k]), ""),
                    "Vmag": float(np.nanmedian(mag[seg, k])),
                    "epoch_start_kyr": float(min(epochs[seg[0]], epochs[seg[-1]])),
                    "epoch_end_kyr": float(max(epochs[seg[0]], epochs[seg[-1]])),
                    "duration_kyr": float(span),
                    "epoch_best_kyr": float(epochs[j]),
                    "azimuth_deg": float(rise_azimuth(dec[j, k], args.lat)),
                })
    mt = pd.DataFrame(match_rows)
    if not mt.empty:
        mt = mt.sort_values(["body", "epoch_start_kyr"])
        mt.to_csv(outdir / "marker_stars.csv", index=False, float_format="%.4f")
        log("")
        log(f"  Stelle V<{args.vmax} che sorgono dove si ferma un corpo "
            f"(entro {args.tol}° di declinazione):")
        for name in bodies:
            sel = mt[mt.body == name]
            if sel.empty:
                continue
            log(f"    {name}:")
            for _, r in sel.sort_values("duration_kyr", ascending=False).head(6).iterrows():
                log(f"      {r['star']:16s} V{r['Vmag']:4.1f} | "
                    f"{r['epoch_start_kyr']:+7.1f} .. {r['epoch_end_kyr']:+7.1f} kyr "
                    f"({r['duration_kyr']:5.1f}) | az {r['azimuth_deg']:5.1f}°")
    else:
        log("  Nessuna stella abbastanza brillante condivide un punto d'arresto.")

    # ---- Milky Way -------------------------------------------------------
    lst = np.arange(0.0, 360.0, 5.0)
    tilt = np.empty((epochs.size, lst.size))
    for j, s in enumerate(lst):
        tilt[:, j] = milky_way_tilt(gp_ra, gp_dec, args.lat, s)
    mw = pd.DataFrame({
        "epoch_kyr": np.repeat(epochs, lst.size),
        "lst_deg": np.tile(lst, epochs.size),
        "milkyway_tilt_deg": tilt.ravel(),
    })
    mw.to_csv(outdir / "milkyway.csv", index=False, float_format="%.3f")
    log("")
    log(f"  Via Lattea: inclinazione sul piano dell'orizzonte fra "
        f"{tilt.min():.1f}° e {tilt.max():.1f}° "
        f"(0° = fascia verticale per lo zenit, 90° = distesa lungo l'orizzonte)")

    # ---- figures ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    style = {"sole_solstizio_estivo": ("darkorange", "-"),
             "sole_solstizio_invernale": ("darkorange", "--"),
             "luna_lunistizio_maggiore_N": ("navy", "-"),
             "luna_lunistizio_maggiore_S": ("navy", "--"),
             "luna_lunistizio_minore_N": ("teal", "-"),
             "luna_lunistizio_minore_S": ("teal", "--")}
    for name in bodies:
        s = bodies_df[bodies_df.body == name]
        c, ls = style[name]
        ax.plot(s["epoch_kyr"], s["rise_azimuth_deg"], color=c, ls=ls, lw=1.4,
                label=name.replace("_", " "))
    if not mt.empty:
        ax.scatter(mt["epoch_best_kyr"], mt["azimuth_deg"], s=14, c="crimson",
                   zorder=5, label=f"stella che tiene il punto (V<{args.vmax})")
    ax.axhline(90.0, color="k", lw=0.7, ls=":")
    ax.text(epochs.min(), 90.5, "est", fontsize=8)
    ax.set_xlabel("Epoca [kyr dall'anno 0]")
    ax.set_ylabel("azimut del sorgere [° da nord]")
    ax.set_title(f"Punti d'arresto sull'orizzonte a {args.lat:+.1f}°, e le stelle "
                 f"che li segnano")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "horizon_points.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    mesh = ax.pcolormesh(epochs, lst, tilt.T, cmap="magma", shading="auto")
    fig.colorbar(mesh, ax=ax,
                 label="inclinazione del piano galattico sull'orizzonte [°]")
    ax.set_xlabel("Epoca [kyr dall'anno 0]")
    ax.set_ylabel("tempo siderale locale [°]")
    ax.set_title(f"Via Lattea a {args.lat:+.1f}°: 0° la fascia sale verticale "
                 f"per lo zenit, 90° giace lungo l'orizzonte")
    fig.tight_layout()
    fig.savefig(outdir / "milkyway.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    log("")
    log(f"Scritti {outdir}/bodies.csv, marker_stars.csv, milkyway.csv")
    log(f"        {outdir}/horizon_points.png, milkyway.png")


if __name__ == "__main__":
    main()
