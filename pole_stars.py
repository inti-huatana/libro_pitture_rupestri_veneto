#!/usr/bin/env python3
"""
pole_stars.py

How well the sky let anyone find north, and south, over the span simulated.

Listing the stars nearest the pole answers the wrong question. What matters is
the bearing a person could actually take, and that has a direct measure: a star
at angular distance rho from the celestial pole turns on a circle of that
radius, so anyone using it as a marker is wrong by up to rho over the course of
a night. The pole distance of the best usable star is therefore the orientation
error itself, in degrees, and needs no invented score weighing brightness
against proximity.

Brightness enters as a threshold rather than as a term to be weighed: a star has
to be seen and picked out before it can be pointed at. The whole analysis is run
at several limits at once -- first magnitude, second, third, fourth -- so the
trade-off is displayed instead of resolved by a coefficient. A dim star very
near the pole and a bright one further out appear as different curves, and the
reader decides which mattered to the people concerned.

Three things come out of it.

    accuracy     the best bearing obtainable at each epoch, per magnitude
                 limit, for both poles. This is the headline: it says when the
                 sky offered a good marker and when it offered none.

    succession   which star held the position at each epoch, and for how long.
                 Reigns are what the question usually means in practice.

    frontier     at chosen epochs, the stars no other star beats on both counts
                 at once. Parameter-free, since domination needs no weights,
                 and it shows what the alternatives were.

Latitude is left out on purpose. A star rho degrees from the pole is
circumpolar wherever the latitude exceeds rho, and any star worth the name has
a small rho, so it is circumpolar across essentially the whole hemisphere. The
answer would be the same at every inhabited latitude, and pretending otherwise
would add a parameter that changes nothing.

Usage
-----
    python3 pole_stars.py --trajectories BSGRID/trajectories.csv --outdir pole
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

# Magnitude limits the whole analysis is repeated at, rather than choosing one.
MAG_LIMITS = (1.5, 2.0, 3.0, 4.0)

# Bearing errors quoted in the coverage table, in degrees. A degree is about two
# solar diameters and near the limit of unaided pointing; ten degrees is a hand's
# breadth at arm's length and no longer a fixed mark.
ERROR_LEVELS = (1.0, 2.0, 5.0, 10.0)


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


def star_labels(traj: pd.DataFrame) -> dict[int, str]:
    lab = traj.groupby("HIP")[["NAME", "Bayer"]].first()
    out = {}
    for hip, r in lab.iterrows():
        name, bayer = str(r["NAME"]).strip(), str(r["Bayer"]).strip()
        out[int(hip)] = (name if name and name != "nan"
                         else bayer if bayer and bayer != "nan"
                         else f"HIP {int(hip)}")
    return out


def best_per_epoch(rho: np.ndarray, mag: np.ndarray, limit: float):
    """Smallest pole distance among stars within the magnitude limit.

    Returns the distance and the column index of the star holding it, with NaN
    and -1 where no star qualifies at all.
    """
    masked = np.where(mag <= limit, rho, np.inf)
    j = np.argmin(masked, axis=1)
    best = masked[np.arange(masked.shape[0]), j]
    ok = np.isfinite(best)
    return np.where(ok, best, np.nan), np.where(ok, j, -1)


def reigns(epochs: np.ndarray, holder: np.ndarray, dist: np.ndarray,
           hips: np.ndarray, labels: dict[int, str], min_kyr: float):
    """Contiguous stretches over which one star stays the nearest.

    Short interruptions are not merged: if the title changes hands and comes
    back, that is what happened, and smoothing it would invent a continuity the
    sky did not have.
    """
    out = []
    n = len(epochs)
    i = 0
    while i < n:
        h = holder[i]
        k = i
        while k + 1 < n and holder[k + 1] == h:
            k += 1
        if h >= 0:
            span = abs(epochs[k] - epochs[i])
            if span >= min_kyr:
                seg = dist[i:k + 1]
                out.append({
                    "HIP": int(hips[h]),
                    "star": labels.get(int(hips[h]), f"HIP {int(hips[h])}"),
                    "epoch_start_kyr": float(min(epochs[i], epochs[k])),
                    "epoch_end_kyr": float(max(epochs[i], epochs[k])),
                    "duration_kyr": float(span),
                    "min_dist_deg": float(np.nanmin(seg)),
                    "epoch_closest_kyr": float(epochs[i + int(np.nanargmin(seg))]),
                })
        i = k + 1
    return out


def pareto_front(rho: np.ndarray, mag: np.ndarray):
    """Indices of stars beaten by no other on both pole distance and magnitude.

    Domination is a comparison, not a sum, so this needs no relative weight
    between the two and is the honest way to show the alternatives.
    """
    order = np.argsort(rho)
    front, best_mag = [], np.inf
    for i in order:
        if mag[i] < best_mag:
            front.append(int(i))
            best_mag = mag[i]
    return front


def main() -> None:
    p = argparse.ArgumentParser(
        description="Orientation quality of the celestial poles through time.")
    p.add_argument("--trajectories", required=True,
                   help="trajectories.csv from bright_star_grid.py")
    p.add_argument("--outdir", default="pole")
    p.add_argument("--Tmin", type=float, default=None)
    p.add_argument("--Tmax", type=float, default=None)
    p.add_argument("--epoch-step", type=float, default=None,
                   help="subsample the file's epochs to this spacing, kyr")
    p.add_argument("--mag-limits", type=float, nargs="+", default=list(MAG_LIMITS))
    p.add_argument("--reign-mag", type=float, default=2.5,
                   help="magnitude limit used for the succession timeline")
    p.add_argument("--min-reign", type=float, default=0.5,
                   help="shortest reign reported, kyr")
    p.add_argument("--frontier-epochs", type=float, nargs="+",
                   default=[0.0, -5.0, -13.0, -26.0, -50.0, -100.0],
                   help="epochs at which the trade-off is tabulated")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log(f"Reading {args.trajectories} ...")
    traj = read_star_table(args.trajectories,
                           ["epoch_kyr_from_year0", "HIP", "dec_deg", "Vmag",
                            "NAME", "Bayer"])
    traj = normalise_epoch(traj).drop_duplicates(["HIP", "epoch"])
    labels = star_labels(traj)

    dec_w = traj.pivot(index="epoch", columns="HIP", values="dec_deg").sort_index()
    mag_w = traj.pivot(index="epoch", columns="HIP", values="Vmag").sort_index()
    epochs = dec_w.index.to_numpy(dtype=float)
    hips = dec_w.columns.to_numpy()
    dec = dec_w.to_numpy(dtype=float)
    mag = mag_w.to_numpy(dtype=float)

    keep = np.ones(epochs.size, dtype=bool)
    if args.Tmin is not None:
        keep &= epochs >= args.Tmin
    if args.Tmax is not None:
        keep &= epochs <= args.Tmax
    if args.epoch_step is not None and epochs.size > 1:
        base = np.min(np.diff(np.sort(epochs)))
        stride = max(1, int(round(args.epoch_step / base)))
        sel = np.zeros(epochs.size, dtype=bool)
        sel[::stride] = True
        keep &= sel
    epochs, dec, mag = epochs[keep], dec[keep], mag[keep]
    log(f"  {hips.size} stars, {epochs.size} epochs "
        f"[{epochs.min():+.1f}, {epochs.max():+.1f}] kyr")

    rho = {"nord": 90.0 - dec, "sud": 90.0 + dec}

    # ---- accuracy ------------------------------------------------------
    acc_rows = []
    curves: dict[tuple[str, float], np.ndarray] = {}
    for pole, r in rho.items():
        for lim in args.mag_limits:
            d, j = best_per_epoch(r, mag, lim)
            curves[(pole, lim)] = d
            for i in range(epochs.size):
                acc_rows.append({
                    "epoch_kyr": epochs[i], "pole": pole, "mag_limit": lim,
                    "best_dist_deg": d[i],
                    "star": (labels.get(int(hips[j[i]]), "") if j[i] >= 0 else ""),
                })
    pd.DataFrame(acc_rows).to_csv(outdir / "accuracy.csv", index=False,
                                  float_format="%.4f")

    # ---- coverage: how much of the span offered what accuracy -----------
    log("")
    log("  Frazione del periodo in cui il polo offriva un riferimento entro")
    log("  un dato errore, per limite di magnitudine:")
    cov_rows = []
    header = "    " + f"{'polo':5s} {'mag':>5s} " + " ".join(
        f"{f'<{e:.0f}°':>7s}" for e in ERROR_LEVELS)
    log(header)
    for pole in rho:
        for lim in args.mag_limits:
            d = curves[(pole, lim)]
            fr = [float(np.nanmean(d <= e)) if np.isfinite(d).any() else 0.0
                  for e in ERROR_LEVELS]
            for e, f in zip(ERROR_LEVELS, fr):
                cov_rows.append({"pole": pole, "mag_limit": lim,
                                 "error_deg": e, "fraction": f})
            log("    " + f"{pole:5s} {lim:5.1f} " +
                " ".join(f"{f:6.0%} " for f in fr))
    pd.DataFrame(cov_rows).to_csv(outdir / "coverage.csv", index=False,
                                  float_format="%.4f")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, pole in zip(axes, ("nord", "sud")):
        for lim in args.mag_limits:
            ax.plot(epochs, curves[(pole, lim)], lw=1.3, label=f"V < {lim:.1f}")
        for e in ERROR_LEVELS:
            ax.axhline(e, color="k", lw=0.5, ls=":")
        ax.set_ylabel(f"errore di orientamento\nal polo {pole} [°]")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=len(args.mag_limits))
    axes[1].set_xlabel("Epoca [kyr dall'anno 0]")
    axes[0].set_title("Migliore riferimento polare disponibile: la distanza dal "
                      "polo\ndella stella piu' vicina entro il limite di "
                      "magnitudine e' l'errore di direzione")
    fig.tight_layout()
    fig.savefig(outdir / "accuracy.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- succession -----------------------------------------------------
    all_reigns = []
    for pole, r in rho.items():
        d, j = best_per_epoch(r, mag, args.reign_mag)
        for row in reigns(epochs, j, d, hips, labels, args.min_reign):
            row["pole"] = pole
            all_reigns.append(row)
    rg = pd.DataFrame(all_reigns)
    if not rg.empty:
        rg = rg.sort_values(["pole", "epoch_start_kyr"])
        rg.to_csv(outdir / "succession.csv", index=False, float_format="%.3f")

        log("")
        log(f"  Successione al polo, stelle piu' luminose di V={args.reign_mag}:")
        for pole in ("nord", "sud"):
            sel = rg[rg["pole"] == pole]
            if sel.empty:
                log(f"    {pole}: nessuna stella entro il limite in tutto il periodo")
                continue
            log(f"    polo {pole}:")
            for _, s in sel.iterrows():
                log(f"      {s['star']:16s} da {s['epoch_start_kyr']:+7.1f} a "
                    f"{s['epoch_end_kyr']:+7.1f} kyr  ({s['duration_kyr']:5.1f} kyr) "
                    f"| minimo {s['min_dist_deg']:5.2f}° a "
                    f"{s['epoch_closest_kyr']:+7.1f}")

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        for ax, pole in zip(axes, ("nord", "sud")):
            sel = rg[rg["pole"] == pole]
            names = list(dict.fromkeys(sel["star"]))
            for _, s in sel.iterrows():
                y = names.index(s["star"])
                ax.barh(y, s["epoch_end_kyr"] - s["epoch_start_kyr"],
                        left=s["epoch_start_kyr"], height=0.6,
                        color=plt.cm.viridis(1.0 - min(s["min_dist_deg"] / 20, 1)))
                ax.text(s["epoch_closest_kyr"], y, f" {s['min_dist_deg']:.1f}°",
                        va="center", fontsize=7)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=8)
            ax.set_ylabel(f"polo {pole}")
            ax.grid(axis="x", alpha=0.3)
        axes[1].set_xlabel("Epoca [kyr dall'anno 0]")
        axes[0].set_title(f"Chi teneva il polo, fra le stelle V < {args.reign_mag}\n"
                          f"(colore e etichetta: minima distanza raggiunta)")
        fig.tight_layout()
        fig.savefig(outdir / "succession.pdf", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ---- trade-off at chosen epochs -------------------------------------
    fr_rows = []
    for t in args.frontier_epochs:
        i = int(np.argmin(np.abs(epochs - t)))
        for pole, r in rho.items():
            good = np.isfinite(r[i]) & np.isfinite(mag[i]) & (r[i] >= 0)
            idx = np.flatnonzero(good)
            if idx.size == 0:
                continue
            front = pareto_front(r[i][idx], mag[i][idx])
            for k in front:
                s = idx[k]
                fr_rows.append({
                    "epoch_kyr": epochs[i], "pole": pole,
                    "HIP": int(hips[s]),
                    "star": labels.get(int(hips[s]), ""),
                    "pole_dist_deg": float(r[i][s]),
                    "Vmag": float(mag[i][s]),
                })
    if fr_rows:
        pd.DataFrame(fr_rows).to_csv(outdir / "frontier.csv", index=False,
                                     float_format="%.3f")
        log("")
        log("  Frontiera del compromesso: stelle che nessun'altra batte su")
        log("  entrambi i fronti, distanza dal polo e magnitudine.")
        fr = pd.DataFrame(fr_rows)
        for t in sorted(fr["epoch_kyr"].unique(), reverse=True):
            for pole in ("nord", "sud"):
                s = fr[(fr["epoch_kyr"] == t) & (fr["pole"] == pole)]
                if s.empty:
                    continue
                txt = ", ".join(f"{r['star']} ({r['pole_dist_deg']:.1f}°, "
                                f"V{r['Vmag']:.1f})" for _, r in s.head(4).iterrows())
                log(f"    {t:+7.1f} kyr {pole:5s}: {txt}")

    log("")
    log(f"Written {outdir}/accuracy.csv, coverage.csv, succession.csv, "
        f"frontier.csv")
    log(f"        {outdir}/accuracy.pdf, succession.pdf")


if __name__ == "__main__":
    main()
