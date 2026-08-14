#!/usr/bin/env python3
"""
canon_epoch.py

When could each sky culture's canon have been assembled, given where it lived.

The observing latitude is taken as known from ethnography, not fitted. A star of
declination d culminates at h = 90 - |phi - d|, and counts as usable when it
clears five degrees: below that, atmospheric extinction, haze and ground relief
put it out of reach of anyone building a figure out of it. The canon is
observable at an epoch when every star it names clears that height, and
precession moves declinations, so the set of such epochs is bounded.

The estimate is therefore an interval, not a point. The count of unusable stars
is an integer and stays at zero over a stretch of epochs; saying the canon was
assembled at one instant inside it would be inventing precision. The result worth
having is when that interval excludes the present, because a modern reconstruction
cannot manufacture that: whoever compiled it worked from today's sky and would
not have added stars invisible from the culture's home now.

Each edge of the interval is set by a single star, the first to drop below five
degrees, so one misattributed identifier in the source would move it. Two things
guard against that: the interval is reported for zero, one, two and three
tolerated stars, and the binding star at each edge is named, which is the part
that can be checked by hand against the ethnographic source.

Usage
-----
    python3 canon_epoch.py \
        --trajectories BSGRID/trajectories.csv \
        --skycultures skycultures_csv \
        --outdir epoch

Output
------
    summary.csv              one row per culture: the interval at each tolerance,
                             whether the present falls inside, and the stars
                             binding each edge

    series/<culture>.csv     unusable count and lowest culmination altitude,
                             epoch by epoch, with the lowest star named

    series/<culture>.pdf     the same drawn, with the five-degree line and the
                             present marked
"""

from __future__ import annotations

import argparse
import fnmatch
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Minimum culmination altitude for a star to be usable, in degrees. Fixed, not
# fitted: five degrees is ample for any culture, and leaving it free was what
# made every earlier version of this analysis unstable.
H_CRIT = 5.0

# Mean obliquity, degrees. Sets how far precession can carry a declination, and
# so which stars are beyond reach from a given latitude at every epoch.
OBLIQUITY_DEG = 23.4

# Tolerated unusable stars, guarding the interval against a misattributed
# identifier in the source reconstruction.
LEVELS = (0, 1, 2, 3)

DEFAULT_PATTERNS = (
    "arabic*", "aztec", "babylonian*", "chinese*", "egyptian*", "greek*",
    "indian*", "hawaiian_starlines", "japanese*", "korean*", "navajo",
    "norse*", "tibetan",
)

# Approximate observing latitude, degrees, positive north. Coarse centroids of
# the region each culture is attached to; override with --latitudes.
DEFAULT_LATITUDES: dict[str, float] = {
    "arabic": 24.0, "arabic_al-sufi": 32.6, "arabic_arabian_peninsula": 24.0,
    "arabic_indigenous": 24.0, "arabic_lunar_stations": 24.0,
    "aztec": 19.4,
    "babylonian_mulapin": 32.5, "babylonian_seleucid": 32.5,
    "chinese": 34.3, "chinese_chenzhuo": 34.3, "chinese_contemporary": 34.3,
    "chinese_medieval": 34.3, "chinese_song_dynasty": 34.3,
    "egyptian": 25.7, "egyptian_dendera": 26.1,
    "greek_almagest": 31.2, "greek_dante": 43.8, "greek_farnese": 31.2,
    "greek_leidenAratea": 31.2, "almagest": 31.2,
    "indian": 25.0, "indian_nakshatras": 25.0,
    "hawaiian_starlines": 20.8,
    "japanese_moon_stations": 35.0,
    "korean": 37.5,
    "navajo": 36.1,
    "norse": 60.0, "norse_edda": 64.0,
    "tibetan": 29.7,
}


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


def matches(name: str, patterns) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def interval_at_level(n_unusable: np.ndarray, epochs: np.ndarray, level: int):
    """Epochs tolerating at most `level` unusable stars.

    Reports the span and whether it is a single stretch: precession is periodic,
    so a long enough scan can admit the canon in more than one window, and a
    plain first-to-last summary would silently bridge the gap between them.
    """
    ok = n_unusable <= level
    if not ok.any():
        return None
    idx = np.flatnonzero(ok)
    return {
        "epoch_lo": float(epochs[idx[0]]),
        "epoch_hi": float(epochs[idx[-1]]),
        "n_epochs": int(ok.sum()),
        "contiguous": bool(idx.size == idx[-1] - idx[0] + 1),
    }


def plot_band(culture, epochs, lat_grid, n_unusable, lat_nom, present_epoch,
              path: Path) -> None:
    """Unusable count over the latitude band, with the nominal latitude marked.

    Drawn as a map rather than reduced to its best cell: the band measures how
    robust the window at the nominal latitude is, it does not replace it.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    mesh = ax.pcolormesh(epochs, lat_grid, np.clip(n_unusable, 0, 20).T,
                         cmap="inferno_r", shading="auto", vmin=0, vmax=20)
    fig.colorbar(mesh, ax=ax, label=f"stelle sotto {H_CRIT:.0f}°")

    levels = [l for l in LEVELS
              if (n_unusable <= l).any() and (n_unusable > l).any()]
    if levels:
        cs = ax.contour(epochs, lat_grid, n_unusable.T,
                        levels=[l + 0.5 for l in levels],
                        colors="cyan", linewidths=1.2)
        ax.clabel(cs, fmt={l + 0.5: str(l) for l in levels}, fontsize=7)

    ax.axhline(lat_nom, color="lime", lw=1.6, ls="--",
               label=f"latitudine nominale {lat_nom:+.1f}°")
    ax.axvline(present_epoch, color="white", lw=1.0, ls=":", label="presente")
    ax.set_xlabel("Epoca [kyr dall'anno 0]")
    ax.set_ylabel("Latitudine dell'osservatore [°]")
    ax.set_title(f"{culture} — stelle inutilizzabili su banda di latitudine")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_series(culture, epochs, n_unusable, h_low, lat, present_epoch,
                path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.step(epochs, n_unusable, where="mid", lw=1.4)
    ax1.axhline(0, color="green", lw=1.0, ls="--", label="canone interamente usabile")
    ax1.axvline(present_epoch, color="k", lw=1.0, ls=":", label="presente")
    ax1.set_ylabel(f"stelle sotto {H_CRIT:.0f}°")
    ax1.set_ylim(bottom=0)
    ax1.set_title(f"{culture} — latitudine {lat:+.1f}°")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, h_low, lw=1.4, color="darkgreen")
    ax2.axhline(H_CRIT, color="crimson", lw=1.2, ls="--",
                label=f"criterio {H_CRIT:.0f}°")
    ax2.axhline(0.0, color="k", lw=0.8)
    ax2.axvline(present_epoch, color="k", lw=1.0, ls=":")
    ax2.set_xlabel("Epoca [kyr dall'anno 0]")
    ax2.set_ylabel("Altezza della stella più bassa [°]")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Epoch interval in which each canon was fully observable.")
    p.add_argument("--trajectories", required=True,
                   help="trajectories.csv from bright_star_grid.py")
    p.add_argument("--skycultures", required=True,
                   help="directory of CSV tables from stellarium_skycultures.py")
    p.add_argument("--outdir", default="epoch")
    p.add_argument("--cultures", nargs="+", default=list(DEFAULT_PATTERNS),
                   help="glob patterns selecting the cultures to analyse")
    p.add_argument("--latitudes", default=None,
                   help="CSV with columns culture,latitude overriding the built-in table")
    p.add_argument("--Tmin", type=float, default=-30.0, help="earliest epoch, kyr")
    p.add_argument("--Tmax", type=float, default=2.0, help="latest epoch, kyr")
    p.add_argument("--epoch-step", type=float, default=0.1, help="epoch step, kyr")
    p.add_argument("--present-kyr", type=float, default=2.0,
                   help="epoch taken as the present, kyr from year 0")
    p.add_argument("--min-stars", type=int, default=10,
                   help="skip cultures with fewer catalogued stars")
    p.add_argument("--lat-halfwidth", type=float, default=20.0,
                   help="half-width of the latitude band scanned around the "
                        "nominal value, degrees")
    p.add_argument("--lat-step", type=float, default=1.0,
                   help="latitude step inside the band, degrees")
    args = p.parse_args()

    outdir = Path(args.outdir)
    (outdir / "series").mkdir(parents=True, exist_ok=True)
    (outdir / "maps").mkdir(parents=True, exist_ok=True)

    latitudes = dict(DEFAULT_LATITUDES)
    if args.latitudes:
        ext = pd.read_csv(args.latitudes)
        latitudes.update(dict(zip(ext["culture"], ext["latitude"].astype(float))))
        log(f"Latitude table overridden for {len(ext)} cultures")

    log(f"Reading {args.trajectories} ...")
    traj = pd.read_csv(args.trajectories,
                       usecols=["epoch_kyr_from_year0", "HIP", "dec_deg",
                                "ecl_lat_deg", "Bayer", "NAME"])
    traj = traj.rename(columns={"epoch_kyr_from_year0": "epoch"})
    traj["epoch"] = traj["epoch"].round(6)
    log(f"  {len(traj):,} rows, {traj['HIP'].nunique()} stars")

    # One label per star, for naming whichever one binds an interval edge.
    lab = traj.groupby("HIP")[["NAME", "Bayer"]].first()
    labels = {}
    for hip, r in lab.iterrows():
        name, bayer = str(r["NAME"]).strip(), str(r["Bayer"]).strip()
        if name and name != "nan":
            labels[hip] = name
        elif bayer and bayer != "nan":
            labels[hip] = bayer
        else:
            labels[hip] = f"HIP {int(hip)}"

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
    log(f"Epochs: {epochs.size} from {epochs.min():+.1f} to {epochs.max():+.1f} kyr")

    # Ecliptic latitude is very nearly invariant under precession, so one value
    # per star suffices. It tells whether a permanently unusable star sits near
    # the south ecliptic pole, which is where the constellations instituted by
    # Keyser and de Houtman in 1598 and by Lacaille in the 1750s live, and so
    # whether its presence in an ancient canon is a modern contamination.
    ecl = traj.pivot(index="epoch", columns="HIP",
                     values="ecl_lat_deg").sort_index().to_numpy(dtype=float)

    present_epoch = float(epochs[np.argmin(np.abs(epochs - args.present_kyr))])
    i_present = int(np.argmin(np.abs(epochs - present_epoch)))
    log(f"Present taken as epoch {present_epoch:+.1f} kyr | criterio {H_CRIT:.0f}°")

    members = pd.read_csv(Path(args.skycultures) / "members.csv")
    all_cultures = sorted(members["culture"].unique())
    cultures = [c for c in all_cultures if matches(c, args.cultures)]
    log(f"Cultures selected: {len(cultures)} of {len(all_cultures)}")
    log("")

    rows = []
    never_rows = []
    for culture in cultures:
        lat = latitudes.get(culture)
        if lat is None:
            log(f"  {culture:28s} SKIP: latitudine non nota")
            continue

        hips = np.sort(members.loc[members["culture"] == culture, "HIP"].unique())
        mask = np.isin(all_hips, hips)
        n_canon = int(mask.sum())
        if n_canon < args.min_stars:
            log(f"  {culture:28s} SKIP: {n_canon} stelle nel catalogo")
            continue

        hips_here = all_hips[mask]
        h = 90.0 - np.abs(lat - dec_all[:, mask])
        n_unusable = (h <= H_CRIT).sum(axis=1)
        j_low = np.argmin(h, axis=1)
        h_low = h[np.arange(h.shape[0]), j_low]
        low_hip = hips_here[j_low]

        pd.DataFrame({
            "epoch_kyr": epochs,
            "n_unusable": n_unusable,
            "h_low_deg": h_low,
            "lowest_HIP": low_hip,
            "lowest_star": [labels.get(hp, f"HIP {int(hp)}") for hp in low_hip],
        }).to_csv(outdir / "series" / f"{culture}.csv", index=False,
                  float_format="%.3f")

        plot_series(culture, epochs, n_unusable, h_low, lat, present_epoch,
                    outdir / "series" / f"{culture}.pdf")

        # Stars that never clear the criterion at the nominal latitude, at any
        # epoch scanned. Their highest culmination and their ecliptic latitude
        # are what distinguish a misattributed identifier from a whole southern
        # block that no northern observer could ever have seen.
        h_max_star = h.max(axis=0)
        never = np.flatnonzero(h_max_star <= H_CRIT)
        for s in never:
            never_rows.append({
                "culture": culture,
                "lat_deg": lat,
                "HIP": int(hips_here[s]),
                "star": labels.get(hips_here[s], f"HIP {int(hips_here[s])}"),
                "ecl_lat_deg": float(ecl[i_present, mask][s]),
                "max_h_deg": float(h_max_star[s]),
                "epoch_of_max_kyr": float(epochs[int(np.argmax(h[:, s]))]),
            })

        # Latitude band: the same count over a window around the nominal value.
        # It separates a wrong latitude, where a nearby one opens a window, from
        # a contaminated reconstruction, where none does.
        lat_grid = np.arange(lat - args.lat_halfwidth,
                             lat + args.lat_halfwidth + 1e-9, args.lat_step)
        n_band = np.empty((epochs.size, lat_grid.size), dtype=np.int32)
        for j, phi in enumerate(lat_grid):
            n_band[:, j] = (90.0 - np.abs(phi - dec_all[:, mask])
                            <= H_CRIT).sum(axis=1)

        pd.DataFrame({
            "epoch_kyr": np.repeat(epochs, lat_grid.size),
            "lat_deg": np.tile(lat_grid, epochs.size),
            "n_unusable": n_band.ravel(),
        }).to_csv(outdir / "maps" / f"{culture}.csv", index=False,
                  float_format="%.3f")
        plot_band(culture, epochs, lat_grid, n_band, lat, present_epoch,
                  outdir / "maps" / f"{culture}.pdf")

        band_min = int(n_band.min())
        has_window = (n_band == 0).any(axis=0)
        lat_ok = lat_grid[has_window]
        bj = int(np.argmin(n_band.min(axis=0)))

        row = {
            "culture": culture,
            "lat_deg": lat,
            "n_stars_culture": len(hips),
            "n_stars_present": n_canon,
            "n_unusable_now": int(n_unusable[i_present]),
            "h_low_now_deg": float(h_low[i_present]),
            "lowest_star_now": labels.get(low_hip[i_present]),
            "min_unusable": int(n_unusable.min()),
            # Softer than the interval and continuous: the epoch at which the
            # marginal star stands highest. Reported as an indication of where
            # inside the interval the canon sat most comfortably, not as a date.
            "epoch_max_margin_kyr": float(epochs[int(np.argmax(h_low))]),
            "max_h_low_deg": float(h_low.max()),
            "n_never_usable": int(never.size),
            # Band diagnostics.
            "band_min_unusable": band_min,
            "band_lat_best_deg": float(lat_grid[bj]),
            "band_lat_window_lo": float(lat_ok.min()) if lat_ok.size else np.nan,
            "band_lat_window_hi": float(lat_ok.max()) if lat_ok.size else np.nan,
            "band_lat_window_width": (float(lat_ok.max() - lat_ok.min())
                                      if lat_ok.size else 0.0),
        }

        for level in LEVELS:
            iv = interval_at_level(n_unusable, epochs, level)
            pre = f"lev{level}"
            if iv is None:
                row.update({f"{pre}_lo": np.nan, f"{pre}_hi": np.nan,
                            f"{pre}_width_kyr": np.nan,
                            f"{pre}_n": 0, f"{pre}_contiguous": True,
                            f"{pre}_at_scan_edge": False,
                            f"{pre}_present_inside": False,
                            f"{pre}_excludes_present": False})
            else:
                inside = bool(n_unusable[i_present] <= level)
                # The width is what separates a date from a truism: a window of
                # one millennium places a canon, one of twenty-seven merely
                # rules out the last few centuries.
                width = iv["epoch_hi"] - iv["epoch_lo"]
                at_edge = bool(iv["epoch_lo"] <= epochs[0] + 1e-9
                               or iv["epoch_hi"] >= epochs[-1] - 1e-9)
                row.update({f"{pre}_lo": iv["epoch_lo"], f"{pre}_hi": iv["epoch_hi"],
                            f"{pre}_width_kyr": width,
                            f"{pre}_n": iv["n_epochs"],
                            f"{pre}_contiguous": iv["contiguous"],
                            f"{pre}_at_scan_edge": at_edge,
                            f"{pre}_present_inside": inside,
                            # The configuration a modern reconstruction cannot
                            # have produced: admissible in the past, not now.
                            f"{pre}_excludes_present": bool(not inside)})

        # Which star closes the window at each edge of the strict interval.
        iv0 = interval_at_level(n_unusable, epochs, 0)
        if iv0 is not None:
            ok = np.flatnonzero(n_unusable == 0)
            i_lo, i_hi = ok[0], ok[-1]
            row["binding_star_lo"] = labels.get(low_hip[max(i_lo - 1, 0)])
            row["binding_star_hi"] = labels.get(
                low_hip[min(i_hi + 1, epochs.size - 1)])
        else:
            row["binding_star_lo"] = ""
            row["binding_star_hi"] = ""

        rows.append(row)

        if iv0 is None:
            verdict = (f"mai usabile (min {int(n_unusable.min())}, "
                       f"{int(never.size)} stelle mai sopra il criterio)")
        else:
            w = iv0["epoch_hi"] - iv0["epoch_lo"]
            edge = "*" if row["lev0_at_scan_edge"] else " "
            gap = "" if row["lev0_contiguous"] else " CON BUCO"
            state = ("presente incluso" if row["lev0_present_inside"]
                     else "PRESENTE ESCLUSO")
            verdict = (f"ampiezza {w:5.1f} kyr{edge} "
                       f"[{iv0['epoch_lo']:+6.1f},{iv0['epoch_hi']:+6.1f}] "
                       f"{state}{gap}")
        log(f"  {culture:28s} {n_canon:4d} st | oggi {row['n_unusable_now']:3d} "
            f"inut. | {verdict}")
        if iv0 is not None and not row["lev0_present_inside"]:
            log(f"  {'':28s}      chiude: {row['binding_star_hi']}  |  "
                f"apre: {row['binding_star_lo']}")

    if not rows:
        raise SystemExit("No culture could be analysed.")

    out = pd.DataFrame(rows)
    out.to_csv(outdir / "summary.csv", index=False)

    log("")
    log(f"Written {outdir}/summary.csv  ({len(out)} cultures)")
    log(f"  presente escluso a tolleranza 0: "
        f"{int(out['lev0_excludes_present'].sum())}")
    log(f"  presente escluso anche tollerando 3 stelle: "
        f"{int(out['lev3_excludes_present'].sum())}")
    log(f"  mai interamente usabile: {int((out['lev3_n'] == 0).sum())}")

    if never_rows:
        nv = pd.DataFrame(never_rows).sort_values(["culture", "ecl_lat_deg"])
        nv.to_csv(outdir / "never_usable.csv", index=False, float_format="%.2f")
        log(f"  {outdir}/never_usable.csv  ({len(nv)} stelle in "
            f"{nv['culture'].nunique()} culture)")
        # Precession carries a star's declination up to sin(beta + eps), so it
        # never clears the criterion from latitude phi, at any epoch, once
        #     beta < phi - 90 + H_CRIT - eps
        # Such a star cannot belong to that canon under any choice of epoch, and
        # near the south ecliptic pole it cannot under any northern latitude
        # either: that region holds the constellations instituted after 1598.
        nv["never_at_any_epoch_below_ecl_lat"] = (
            nv["lat_deg"] - 90.0 + H_CRIT - OBLIQUITY_DEG)
        deep = nv[nv["ecl_lat_deg"] < nv["never_at_any_epoch_below_ecl_lat"]]
        log(f"    di cui sotto la soglia di inaccessibilita' permanente "
            f"(beta < lat - 90 + {H_CRIT:.0f} - {OBLIQUITY_DEG:.1f}): {len(deep)}")

    log("")
    log("  Canoni la cui finestra esclude il presente, per ampiezza crescente")
    log("  (ampiezza piccola = datazione stretta; * = bordo sullo scan):")
    strict = out[out["lev0_excludes_present"] & (out["lev0_n"] > 0)]
    if strict.empty:
        log("    nessuno")
    for _, r in strict.sort_values("lev0_width_kyr").iterrows():
        edge = "*" if r["lev0_at_scan_edge"] else " "
        log(f"    {r['culture']:26s} {r['lev0_width_kyr']:6.1f} kyr{edge} "
            f"[{r['lev0_lo']:+6.1f},{r['lev0_hi']:+6.1f}] | "
            f"apre {str(r['binding_star_lo'])[:14]:14s} chiude {r['binding_star_hi']}")


if __name__ == "__main__":
    main()
