#!/usr/bin/env python3
"""
constellation_drift.py

How much each constellation's shape deforms under stellar proper motion.

The deformation is a property of the stars, not of the lines drawn between them:
a constellation only selects which pairs of stars to look at, and the angular
separation of a pair changes by the same amount whoever grouped it with whatever
else. So the figures used here are only a presentation layer, and the modern IAU
ones serve as well as any -- better, in fact, since they are uniform in
construction and carry none of the reconstruction bias of an ethnographic
source. Because --culture takes any sky culture the parser produces, that claim
can be checked rather than assumed: run the same analysis on a second culture
and see whether the ranking moves.

Measurement
-----------
For a figure of n stars, all n(n-1)/2 internal angular separations are computed
at every epoch and divided by their values at the reference epoch. Internal
separations are invariant under precession, which is a rotation of the whole
frame, so what is left is proper motion alone.

Two numbers come out of the ratios, kept apart because they mean different
things:

    scale       their median. A group receding or approaching shrinks or grows
                as a whole without changing shape, and for a bound cluster this
                is nearly all of the effect.

    distortion  the RMS spread of the ratios about that median, in per cent.
                This is shape change proper: the figure ceasing to be the same
                figure. A few per cent is imperceptible, a few tens of per cent
                is a different pattern.

Which star is responsible is found by leaving each one out in turn and taking
the one whose absence most reduces the distortion. What governs it is proper
motion, which grows with nearness but also with the star's own space velocity,
so the nearest member is the usual culprit without being the inevitable one: the
Southern Cross comes apart because of Gacrux at 27 pc, whose removal takes the
figure from 103 per cent distortion to 6 while its other three stars all lie
beyond 85 pc and hold their positions relative to one another, but Cassiopeia is
dismantled by Schedar at 71 pc rather than by Ruchbah at 31. The correlation
between nearness and deformation is therefore computed across the whole set and
printed, instead of being asserted.

No threshold of recognisability is imposed. The epoch at which the distortion
first crosses 5, 10 and 25 per cent is reported, and the reader picks.

Usage
-----
    python3 constellation_drift.py \
        --trajectories fullcat.csv.gz \
        --skycultures skycultures_csv \
        --culture modern_iau \
        --outdir drift
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

# Levels at which the crossing epoch is reported, in per cent of shape change.
CROSS_LEVELS = (5.0, 10.0, 25.0)


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


def separations(ra_deg: np.ndarray, dec_deg: np.ndarray, ia, ib) -> np.ndarray:
    """Angular separation of each pair, per epoch. Shape (n_epoch, n_pair)."""
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    sa, ca = np.sin(dec[:, ia]), np.cos(dec[:, ia])
    sb, cb = np.sin(dec[:, ib]), np.cos(dec[:, ib])
    cosd = sa * sb + ca * cb * np.cos(ra[:, ia] - ra[:, ib])
    return np.degrees(np.arccos(np.clip(cosd, -1.0, 1.0)))


def shape_series(sep: np.ndarray, i_ref: int):
    """Scale factor and shape distortion per epoch, from pairwise separations.

    The median ratio is removed before measuring the spread because a uniform
    expansion leaves a figure the same figure; what is reported as distortion is
    only the part that cannot be undone by rescaling.
    """
    ref = sep[i_ref]
    good = ref > 1e-9
    ratio = sep[:, good] / ref[good]
    scale = np.median(ratio, axis=1)
    resid = ratio / scale[:, None] - 1.0
    return scale, 100.0 * np.sqrt(np.mean(resid ** 2, axis=1))


def crossing_epoch(epochs: np.ndarray, dist: np.ndarray, level: float,
                   i_ref: int):
    """Epoch, going back from the reference, where distortion first exceeds level."""
    order = np.argsort(epochs)[::-1]          # from the reference backwards
    for k in order:
        if epochs[k] <= epochs[i_ref] and dist[k] > level:
            return float(epochs[k])
    return np.nan


def worst_star(ra, dec, idx, i_ref, i_far):
    """The member whose removal most reduces the distortion at the far epoch."""
    if idx.size < 4:
        return None, np.nan
    base_pairs = np.triu_indices(idx.size, k=1)
    base = shape_series(separations(ra, dec, idx[base_pairs[0]],
                                    idx[base_pairs[1]]), i_ref)[1][i_far]
    best, best_drop = None, -np.inf
    for s in range(idx.size):
        keep = np.delete(np.arange(idx.size), s)
        sub = idx[keep]
        ia, ib = np.triu_indices(sub.size, k=1)
        d = shape_series(separations(ra, dec, sub[ia], sub[ib]), i_ref)[1][i_far]
        if base - d > best_drop:
            best_drop, best = base - d, s
    return best, best_drop


def main() -> None:
    p = argparse.ArgumentParser(
        description="Shape deformation of constellations under proper motion.")
    p.add_argument("--trajectories", required=True,
                   help="trajectories.csv or fullcat.csv.gz")
    p.add_argument("--skycultures", required=True,
                   help="directory of CSV tables from stellarium_skycultures.py")
    p.add_argument("--culture", default="modern_iau",
                   help="sky culture supplying the figures; the physics does "
                        "not depend on it, which is checkable by rerunning with "
                        "another one")
    p.add_argument("--outdir", default="drift")
    p.add_argument("--ref-kyr", type=float, default=0.0,
                   help="reference epoch the shapes are compared against")
    p.add_argument("--min-stars", type=int, default=4,
                   help="skip figures with fewer catalogued stars; three stars "
                        "give only three separations and no leave-one-out")
    args = p.parse_args()

    outdir = Path(args.outdir)
    (outdir / "curves").mkdir(parents=True, exist_ok=True)

    log(f"Reading {args.trajectories} ...")
    traj = pd.read_csv(args.trajectories, low_memory=False)
    if "epoch_kyr_from_year0" in traj.columns:
        traj = traj.rename(columns={"epoch_kyr_from_year0": "epoch"})
    elif "epoch_kyr" in traj.columns:
        traj = traj.rename(columns={"epoch_kyr": "epoch"})
    traj["epoch"] = traj["epoch"].round(6)
    traj = traj.drop_duplicates(["HIP", "epoch"])

    ra_w = traj.pivot(index="epoch", columns="HIP", values="ra_deg").sort_index()
    dec_w = traj.pivot(index="epoch", columns="HIP", values="dec_deg").sort_index()
    dist_w = traj.pivot(index="epoch", columns="HIP", values="distance_pc").sort_index()
    epochs = ra_w.index.to_numpy(dtype=float)
    all_hips = ra_w.columns.to_numpy()
    ra = ra_w.to_numpy(dtype=float)
    dec = dec_w.to_numpy(dtype=float)
    dpc = dist_w.to_numpy(dtype=float)
    log(f"  {all_hips.size} stars, {epochs.size} epochs "
        f"[{epochs.min():+.1f}, {epochs.max():+.1f}] kyr")

    i_ref = int(np.argmin(np.abs(epochs - args.ref_kyr)))
    i_far = int(np.argmin(epochs))
    log(f"Reference epoch {epochs[i_ref]:+.1f} kyr | "
        f"farthest {epochs[i_far]:+.1f} kyr")

    labels = {}
    lab = traj.groupby("HIP")[["NAME", "Bayer"]].first()
    for hip, r in lab.iterrows():
        name, bayer = str(r["NAME"]).strip(), str(r["Bayer"]).strip()
        labels[hip] = (name if name and name != "nan"
                       else bayer if bayer and bayer != "nan"
                       else f"HIP {int(hip)}")

    members = pd.read_csv(Path(args.skycultures) / "members.csv")
    cons = pd.read_csv(Path(args.skycultures) / "constellations.csv")
    sub = members[members["culture"] == args.culture]
    if sub.empty:
        avail = ", ".join(sorted(members["culture"].unique())[:12])
        raise SystemExit(f"Culture '{args.culture}' not found. Available: {avail} ...")
    names = dict(zip(cons.loc[cons["culture"] == args.culture, "constellation_id"],
                     cons.loc[cons["culture"] == args.culture, "name_english"]))

    log(f"Culture: {args.culture} | "
        f"{sub['constellation_id'].nunique()} figures")
    log("")

    rows = []
    for cid, grp in sub.groupby("constellation_id"):
        hips = np.sort(grp["HIP"].unique())
        idx = np.flatnonzero(np.isin(all_hips, hips))
        if idx.size < args.min_stars:
            continue

        ia, ib = np.triu_indices(idx.size, k=1)
        sep = separations(ra, dec, idx[ia], idx[ib])
        scale, dist = shape_series(sep, i_ref)

        pd.DataFrame({
            "epoch_kyr": epochs,
            "scale": scale,
            "distortion_pct": dist,
            "mean_sep_deg": sep.mean(axis=1),
        }).to_csv(outdir / "curves" / f"{cid.replace(' ', '_')}.csv",
                  index=False, float_format="%.5f")

        s, drop = worst_star(ra, dec, idx, i_ref, i_far)
        driver = labels.get(all_hips[idx[s]]) if s is not None else ""
        driver_pc = float(dpc[i_ref, idx[s]]) if s is not None else np.nan

        row = {
            "culture": args.culture,
            "constellation_id": cid,
            "name": names.get(cid, ""),
            "n_stars": idx.size,
            "n_stars_figure": len(hips),
            "mean_sep_ref_deg": float(sep[i_ref].mean()),
            "nearest_pc": float(np.nanmin(dpc[i_ref, idx])),
            "median_pc": float(np.nanmedian(dpc[i_ref, idx])),
            "distortion_far_pct": float(dist[i_far]),
            "scale_far": float(scale[i_far]),
            "driver_star": driver,
            "driver_distance_pc": driver_pc,
            "driver_drop_pct": float(drop) if s is not None else np.nan,
        }
        for lv in CROSS_LEVELS:
            row[f"epoch_cross_{lv:.0f}pct"] = crossing_epoch(epochs, dist, lv, i_ref)
        rows.append(row)

    if not rows:
        raise SystemExit("No figure had enough catalogued stars.")

    out = pd.DataFrame(rows).sort_values("distortion_far_pct")
    out.to_csv(outdir / "summary.csv", index=False)

    far = epochs[i_far]
    log(f"  Deformazione a {far:+.0f} kyr, dalle piu' stabili alle piu' alterate:")
    log(f"    {'costellazione':22s} {'n':>3s} {'defor.':>8s} {'piu vicina':>11s} "
        f"{'responsabile':>16s} {'a pc':>7s}")
    for _, r in out.iterrows():
        log(f"    {str(r['name'])[:22]:22s} {r['n_stars']:3d} "
            f"{r['distortion_far_pct']:7.1f}% {r['nearest_pc']:10.0f} "
            f"{str(r['driver_star'])[:16]:>16s} {r['driver_distance_pc']:7.0f}")

    # The mechanism, stated as a number rather than an assertion: if the nearest
    # member governs the deformation, the two must track each other.
    ok = out["nearest_pc"].notna() & out["distortion_far_pct"].notna()
    if ok.sum() > 3:
        rho = np.corrcoef(np.log10(out.loc[ok, "nearest_pc"]),
                          np.log10(out.loc[ok, "distortion_far_pct"] + 1e-3))[0, 1]
        log("")
        log(f"  Correlazione log-log fra distanza della stella piu' vicina e "
            f"deformazione: {rho:+.2f}")
        log(f"    (negativa e forte = la vicinanza di un membro governa la "
            f"deformazione)")

    fig, ax = plt.subplots(figsize=(11, 7))
    for _, r in out.iterrows():
        c = pd.read_csv(outdir / "curves" /
                        f"{r['constellation_id'].replace(' ', '_')}.csv")
        ax.plot(c["epoch_kyr"], c["distortion_pct"], lw=0.8, alpha=0.5)
    for _, r in pd.concat([out.head(3), out.tail(3)]).iterrows():
        c = pd.read_csv(outdir / "curves" /
                        f"{r['constellation_id'].replace(' ', '_')}.csv")
        ax.plot(c["epoch_kyr"], c["distortion_pct"], lw=2.0,
                label=f"{r['name']} ({r['distortion_far_pct']:.0f}%)")
    for lv in CROSS_LEVELS:
        ax.axhline(lv, color="k", lw=0.6, ls=":")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("Epoca [kyr dall'anno 0]")
    ax.set_ylabel("Deformazione di forma [%]")
    ax.set_title(f"{args.culture} — alterazione delle figure per moto proprio")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "drift.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)

    log("")
    log(f"Written {outdir}/summary.csv ({len(out)} figure) e {outdir}/drift.pdf")


if __name__ == "__main__":
    main()
