#!/usr/bin/env python3
"""
constellation_visibility.py

Constrain the observing latitude and the epoch of a sky culture from the
requirement that every star it names was actually visible, estimating the
practical horizon and the outlier rate from the data instead of assuming them.

Supersedes the empty-zone approach of constellation_age.py, which inferred from
an absence -- no constellations near the south celestial pole -- and therefore
could not work for a southern culture at all. This one infers from a presence: a
star belongs to the culture's canon, so somebody saw it, so it rose. That is a
physical constraint rather than a fit to a hole, and it is symmetric in the
hemispheres.

The estimator
-------------
A star of declination d culminates at altitude h = 90 - |phi - d| for an observer
at latitude phi. At a candidate (phi, epoch) every star of the bright-star
catalogue therefore has a known h, and is either in the culture's canon or not.

A culture that observed from that latitude at that epoch should show a cutoff:
above some altitude its stars enter the canon at some rate, below it they stop,
because a star culminating a couple of degrees up carries two magnitudes of
extinction and is useless for building a figure. That cutoff is the practical
horizon. It is not assumed here, it is estimated, together with the rate at
which stars leak in below it, which is what an outlier tolerance really is.

Model: a catalogue star enters the canon with probability p_above when
h >= h_min and p_below when h < h_min. Splitting the catalogue by canon
membership and by the cutoff gives a 2x2 table

                    h >= h_min     h < h_min
    in canon             A              B
    not in canon         C              D

whose log-likelihood is maximised in closed form by p_above = A/(A+C) and
p_below = B/(B+D), so no numerical optimisation is needed and h_min can simply
be scanned. Nothing is chosen by hand:

    h_min     the practical horizon, fitted
    B         how many canon stars sit below it, fitted, not a tolerance
    p_below   the leak rate, the outlier fraction in its natural form

A culture with a real horizon cutoff has p_below near zero and p_above well
above it. A culture whose canon says nothing about visibility has p_below close
to p_above; the likelihood ratio against that null, reported as `lr_cutoff`,
measures how much of a cutoff there is at all. Because h_min is a threshold
parameter the usual chi-square calibration of that ratio does not hold, so it is
reported descriptively and a permutation test is available with --permutations.

Only p_above > p_below is admitted, because a horizon can only suppress. Without
that ordering the likelihood also rewards the reverse arrangement, in which the
canon crowds towards the horizon and everything else stands high: since
h = 90 - |phi - d| is symmetric under reflection, every genuine solution has a
mirror at a latitude that turns it upside down, and placing a northern canon far
enough south produces exactly that. Those mirrors are strong fits to a
physically empty pattern, and the ordering constraint is what removes them.

Two things stay entangled even so. The southern edge of a canon constrains only
phi - h_min; the northern edge would constrain phi + h_min and separate them,
but a culture whose canon reaches the celestial pole has no northern edge, so
latitude and practical horizon slide along one another. The summary therefore
reports phi_minus_hmin as the combination the data actually fix, and h_min is
bounded above at a value taken from atmospheric extinction -- roughly 0.8 mag at
15 degrees of altitude and negligible higher -- rather than left free to absorb
the degeneracy.

Profiling the likelihood over h_min at every (phi, epoch) yields a surface whose
maximum locates the observing latitude and the epoch jointly, and whose
2*Delta-logL contours delimit a confidence region. Precession sweeps
declinations by up to +/-23.4 degrees, so the surface has real structure in
epoch; its period is 25.8 kyr, so within the Holocene the solution is unique.

What to look for
----------------
Modern reconstructions are built by scholars who know where a culture lived and
would not have included stars invisible from there today, so finding a canon
consistent with its own latitude at the present epoch is close to circular. The
informative case is the opposite one, flagged as `informative`:

    the present epoch is excluded at the known latitude, while some past epoch
    is not

which no reconstruction can have manufactured, and which marks a canon that only
makes sense as an inheritance from an earlier sky. Expect few such cultures: for
most, the constraint will be wide and the present perfectly admissible, and that
is the correct answer rather than a failure.

Usage
-----
    python3 constellation_visibility.py \
        --trajectories BSGRID/trajectories.csv \
        --skycultures skycultures_csv \
        --outdir constellation_visibility_out

Output
------
    summary.csv              one row per culture: the maximum, the fitted
                             practical horizon, the two inclusion rates, the
                             95% ranges and the verdict on the present epoch

    surfaces/<culture>.csv   the profile surface in long form, one row per
                             (epoch, latitude), carrying the log-likelihood, its
                             distance from the maximum, whether the cell is
                             inside the 95% region, the cutoff fitted there and
                             the four counts behind it

    surfaces/<culture>.pdf   the same surface drawn, with the known latitude and
                             the present epoch marked

Grid resolution, and so the size of the surface files, is set by --epoch-step
and --lat-step.
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

# Chi-square quantiles used only to draw and describe the regions.
CHI2_1DOF_95 = 3.841
CHI2_2DOF_95 = 5.991

# Approximate observing latitude of each sky culture, in degrees, positive north.
# Coarse centroids of the region a culture is attached to, meant to be reviewed
# and overridden with --latitudes; they carry no better precision than a degree
# or two, and for a seafaring culture a single latitude is a poor model of a
# voyaging range in the first place. A culture absent from this table is still
# analysed, but only its two-dimensional surface is reported, since without a
# latitude there is no epoch constraint to extract.
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
# Likelihood
# ---------------------------------------------------------------------------
def binom_loglik(k, n):
    """Log-likelihood of k successes in n trials at the maximum, p = k/n.

    Zero when n is zero and when k is 0 or n, the conventions that make the
    two halves of the split table combine without special-casing empty halves.
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    m = n - k
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(k > 0, k * np.log(np.where(k > 0, k / np.where(n > 0, n, 1.0), 1.0)), 0.0)
        b = np.where(m > 0, m * np.log(np.where(m > 0, m / np.where(n > 0, n, 1.0), 1.0)), 0.0)
    return np.where(n > 0, a + b, 0.0)


def profile_cutoff(h: np.ndarray, in_canon: np.ndarray, hmin_grid: np.ndarray):
    """Best cutoff altitude for one (latitude, epoch), by profile likelihood.

    Sorting the catalogue by culmination altitude once turns the counts of the
    2x2 table into cumulative sums, so every candidate cutoff is evaluated
    without re-scanning the stars. Returns the maximised log-likelihood, the
    cutoff attaining it, the four counts there, and the log-likelihood of the
    null in which membership does not depend on altitude at all.

    Only cutoffs with p_above > p_below are admitted. A horizon makes stars
    below it *less* likely to be named, and an unsigned likelihood is happy to
    certify the reverse: placing a northern canon at a far southern latitude
    drives all of its stars towards the horizon while the rest of the catalogue
    stays high, which is a strong anti-correlation and would otherwise score as
    a strong fit. That is the mirror solution created by h = 90 - |phi - d|
    being symmetric under reflection, and it has no physical meaning. When no
    cutoff satisfies the ordering the cell simply falls back on the null.
    """
    order = np.argsort(h, kind="stable")
    hs = h[order]
    ins = in_canon[order].astype(np.int64)

    cum_in = np.concatenate(([0], np.cumsum(ins)))
    n = h.size
    total_in = int(cum_in[-1])
    ll_null = float(binom_loglik(total_in, n))

    # Number of catalogue stars strictly below each candidate cutoff.
    idx = np.searchsorted(hs, hmin_grid, side="left")

    B = cum_in[idx].astype(float)          # canon stars below the cutoff
    D = idx.astype(float) - B              # other stars below
    A = float(total_in) - B                # canon stars at or above
    C = (n - idx).astype(float) - A        # other stars at or above

    above = A + C
    below = B + D
    with np.errstate(divide="ignore", invalid="ignore"):
        p_above = np.where(above > 0, A / np.where(above > 0, above, 1.0), np.nan)
        p_below = np.where(below > 0, B / np.where(below > 0, below, 1.0), np.nan)

    ll = binom_loglik(A, above) + binom_loglik(B, below)
    ok = np.isfinite(p_above) & np.isfinite(p_below) & (p_above > p_below)
    if not ok.any():
        return ll_null, np.nan, float(total_in), 0.0, float(n - total_in), 0.0, ll_null

    ll = np.where(ok, ll, -np.inf)
    j = int(np.argmax(ll))
    return (float(ll[j]), float(hmin_grid[j]),
            float(A[j]), float(B[j]), float(C[j]), float(D[j]), ll_null)


def surface(dec_canon: np.ndarray, dec_other: np.ndarray, lat_grid: np.ndarray,
            hmin_grid: np.ndarray):
    """Profile log-likelihood over the (epoch, latitude) grid.

    dec_canon and dec_other hold declinations of date, shaped (n_epoch, n_star),
    for the catalogue stars that are and are not in the culture's canon.
    """
    n_epoch = dec_canon.shape[0]
    n_lat = lat_grid.size

    ll = np.full((n_epoch, n_lat), -np.inf)
    ll_null = np.zeros(n_epoch)
    hmin = np.full((n_epoch, n_lat), np.nan)
    cnt = np.zeros((n_epoch, n_lat, 4))

    in_canon = np.concatenate((np.ones(dec_canon.shape[1], dtype=bool),
                               np.zeros(dec_other.shape[1], dtype=bool)))

    for i in range(n_epoch):
        dec = np.concatenate((dec_canon[i], dec_other[i]))
        for j, phi in enumerate(lat_grid):
            h = 90.0 - np.abs(phi - dec)
            v, hm, A, B, C, D, lln = profile_cutoff(h, in_canon, hmin_grid)
            ll[i, j] = v
            hmin[i, j] = hm
            cnt[i, j] = (A, B, C, D)
            ll_null[i] = lln

    return ll, hmin, cnt, ll_null


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def write_surface_csv(path: Path, epochs, lat_grid, ll, hmin, cnt,
                      ll_best: float) -> int:
    """Write the profile surface in long form, one row per (epoch, latitude).

    Everything the fit produces at a cell is kept, not just the log-likelihood:
    the cutoff attaining it and the four counts behind it are what make a cell
    interpretable, and they are computed anyway.
    """
    n_epoch, n_lat = ll.shape
    ep = np.repeat(epochs, n_lat)
    la = np.tile(lat_grid, n_epoch)

    A = cnt[:, :, 0].ravel()
    B = cnt[:, :, 1].ravel()
    C = cnt[:, :, 2].ravel()
    D = cnt[:, :, 3].ravel()
    above = A + C
    below = B + D

    flat_ll = ll.ravel()
    df = pd.DataFrame({
        "epoch_kyr": ep,
        "lat_deg": la,
        "loglik": flat_ll,
        # Distance from the best cell, in the units the confidence contour uses.
        "delta_2loglik": 2.0 * (ll_best - flat_ll),
        "inside_95": 2.0 * (ll_best - flat_ll) <= CHI2_2DOF_95,
        "h_min_fitted_deg": hmin.ravel(),
        "n_canon_above": A.astype(int),
        "n_canon_below": B.astype(int),
        "n_other_above": C.astype(int),
        "n_other_below": D.astype(int),
        "p_above": np.divide(A, above, out=np.full_like(A, np.nan), where=above > 0),
        "p_below": np.divide(B, below, out=np.full_like(B, np.nan), where=below > 0),
    })
    df.to_csv(path, index=False, float_format="%.4f")
    return len(df)


def plot_culture(culture: str, epochs, lat_grid, ll, lat_known,
                 present_epoch, best, path: Path) -> None:
    dev = 2.0 * (np.nanmax(ll) - ll)

    fig, ax = plt.subplots(figsize=(11, 6))
    mesh = ax.pcolormesh(epochs, lat_grid, np.clip(dev, 0, 40).T,
                         cmap="viridis_r", shading="auto", vmin=0, vmax=40)
    fig.colorbar(mesh, ax=ax, label=r"$2\,\Delta\log L$ dal massimo")

    ax.contour(epochs, lat_grid, dev.T, levels=[CHI2_2DOF_95],
               colors="white", linewidths=1.6)

    ax.plot(best["epoch"], best["lat"], "r*", ms=14, label="massimo")
    if lat_known is not None:
        ax.axhline(lat_known, color="orange", lw=1.4, ls="--",
                   label=f"latitudine nota {lat_known:+.1f}°")
    ax.axvline(present_epoch, color="white", lw=1.0, ls=":", label="presente")

    ax.set_xlabel("Epoca [kyr dall'anno 0]")
    ax.set_ylabel("Latitudine dell'osservatore [°]")
    ax.set_title(f"{culture} — verosimiglianza profilata; "
                 f"contorno bianco = regione al 95%")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Latitude and epoch of a sky culture from canon visibility.")
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
    p.add_argument("--epoch-step", type=float, default=0.2, help="epoch step, kyr")
    p.add_argument("--lat-step", type=float, default=1.0, help="latitude step, deg")
    p.add_argument("--lat-min", type=float, default=-55.0,
                   help="southernmost observer latitude scanned, deg")
    p.add_argument("--lat-max", type=float, default=70.0,
                   help="northernmost observer latitude scanned, deg")
    p.add_argument("--hmin-max", type=float, default=15.0,
                   help="upper bound of the fitted practical horizon, deg; set "
                        "from atmospheric extinction, which is about 0.8 mag at "
                        "15 deg altitude and negligible above it")
    p.add_argument("--hmin-step", type=float, default=0.5,
                   help="resolution of the fitted practical horizon, deg")
    p.add_argument("--present-kyr", type=float, default=2.0,
                   help="epoch taken as the present, kyr from year 0")
    p.add_argument("--min-stars", type=int, default=10,
                   help="skip cultures with fewer catalogued stars")
    p.add_argument("--permutations", type=int, default=0,
                   help="permutation replicas calibrating lr_cutoff (0 skips)")
    p.add_argument("--seed", type=int, default=20260814)
    args = p.parse_args()

    outdir = Path(args.outdir)
    (outdir / "surfaces").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

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
    epochs_sel = keep[::stride]
    traj = traj[traj["epoch"].isin(epochs_sel)]

    wide = traj.pivot(index="epoch", columns="HIP", values="dec_deg").sort_index()
    epochs = wide.index.to_numpy(dtype=float)
    all_hips = wide.columns.to_numpy()
    dec_all = wide.to_numpy(dtype=float)
    log(f"Epochs: {epochs.size} from {epochs.min():+.1f} to {epochs.max():+.1f} kyr")

    # Latitudes are limited to the inhabited range: outside it the fit has no
    # subject, and the far south in particular is where the mirror solutions of
    # a northern canon used to pile up.
    lat_grid = np.arange(args.lat_min, args.lat_max + 1e-9, args.lat_step)
    hmin_grid = np.arange(0.0, args.hmin_max + 1e-9, args.hmin_step)
    log(f"Latitudes: {lat_grid.size} from {lat_grid[0]:+.0f} to {lat_grid[-1]:+.0f} "
        f"| cutoff grid: {hmin_grid.size} (0 .. {args.hmin_max:.0f}°)")

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
        n_present = int(mask.sum())
        if n_present < args.min_stars:
            log(f"  {culture:28s} SKIP: {n_present} stelle nel catalogo")
            continue

        dec_canon = dec_all[:, mask]
        dec_other = dec_all[:, ~mask]

        ll, hmin, cnt, ll_null = surface(dec_canon, dec_other, lat_grid, hmin_grid)

        flat = int(np.argmax(ll))
        bi, bj = np.unravel_index(flat, ll.shape)
        A, B, C, D = cnt[bi, bj]
        best = {"epoch": float(epochs[bi]), "lat": float(lat_grid[bj])}
        lr_cutoff = 2.0 * (ll[bi, bj] - ll_null[bi])

        dev = 2.0 * (ll[bi, bj] - ll)
        inside = dev <= CHI2_2DOF_95
        lat_lo = float(lat_grid[inside.any(axis=0)].min())
        lat_hi = float(lat_grid[inside.any(axis=0)].max())
        ep_lo = float(epochs[inside.any(axis=1)].min())
        ep_hi = float(epochs[inside.any(axis=1)].max())

        lat_known = latitudes.get(culture)
        write_surface_csv(outdir / "surfaces" / f"{culture}.csv",
                          epochs, lat_grid, ll, hmin, cnt, float(ll[bi, bj]))
        plot_culture(culture, epochs, lat_grid, ll, lat_known,
                     present_epoch, best, outdir / "surfaces" / f"{culture}.pdf")

        row = {
            "culture": culture,
            "n_stars_culture": len(hips),
            "n_stars_present": n_present,
            "coverage": n_present / len(hips),
            "best_epoch_kyr": best["epoch"],
            "best_epoch_year": best["epoch"] * 1000.0,
            "best_latitude_deg": best["lat"],
            "fitted_h_min_deg": float(hmin[bi, bj]),
            # The southern edge of the canon constrains only phi - h_min: a
            # northern culture reaching the celestial pole gives no northern
            # edge to break the degeneracy, so the two are traded off along this
            # combination and it, not the latitude alone, is what the data fix.
            "phi_minus_hmin_deg": best["lat"] - float(hmin[bi, bj]),
            "n_canon_above": A, "n_canon_below": B,
            "p_above": A / (A + C) if (A + C) > 0 else np.nan,
            "p_below": B / (B + D) if (B + D) > 0 else np.nan,
            "contrast": ((A / (A + C) if (A + C) > 0 else np.nan)
                         - (B / (B + D) if (B + D) > 0 else np.nan)),
            "lr_cutoff": lr_cutoff,
            "lat_lo95_deg": lat_lo, "lat_hi95_deg": lat_hi,
            "epoch_lo95_kyr": ep_lo, "epoch_hi95_kyr": ep_hi,
            "lat_known_deg": lat_known if lat_known is not None else np.nan,
        }

        if lat_known is None:
            row.update(present_excluded=np.nan, best_epoch_at_known_kyr=np.nan,
                       lr_present=np.nan, informative=False,
                       notes="latitudine non nota: solo superficie 2D")
        else:
            jk = int(np.argmin(np.abs(lat_grid - lat_known)))
            prof = ll[:, jk]
            ib = int(np.argmax(prof))
            lr_present = 2.0 * (prof[ib] - prof[i_present])
            excluded = bool(lr_present > CHI2_1DOF_95)
            row.update(
                best_epoch_at_known_kyr=float(epochs[ib]),
                lr_present=float(lr_present),
                present_excluded=excluded,
                # The one configuration a modern reconstruction cannot produce.
                informative=bool(excluded and np.isfinite(prof[ib])),
                notes="",
            )

        if args.permutations > 0:
            # Membership is shuffled among catalogue stars and the whole
            # maximisation repeated, which calibrates lr_cutoff without relying
            # on a chi-square that the threshold parameter invalidates.
            null_lr = np.empty(args.permutations)
            for r in range(args.permutations):
                perm = rng.permutation(dec_all.shape[1])
                dc = dec_all[:, perm[:n_present]]
                do = dec_all[:, perm[n_present:]]
                llp, _, _, llnp = surface(dc, do, lat_grid, hmin_grid)
                f = int(np.argmax(llp))
                pi, pj = np.unravel_index(f, llp.shape)
                null_lr[r] = 2.0 * (llp[pi, pj] - llnp[pi])
            row["p_lr_permutation"] = float((null_lr >= lr_cutoff).mean())
        else:
            row["p_lr_permutation"] = np.nan

        rows.append(row)

        lat_txt = (f"lat nota {lat_known:+5.1f}°" if lat_known is not None
                   else "lat ignota    ")
        verdict = ("INFORMATIVA" if row["informative"]
                   else ("presente ammesso" if lat_known is not None else "--"))
        log(f"  {culture:28s} {n_present:4d} st | max ({best['epoch']:+6.1f} kyr, "
            f"{best['lat']:+5.1f}°) | h_min {row['fitted_h_min_deg']:4.1f}° | "
            f"p_sotto {row['p_below']:.3f} vs p_sopra {row['p_above']:.3f} | "
            f"LR {lr_cutoff:6.1f} | {lat_txt} | {verdict}")

    if not rows:
        raise SystemExit("No culture could be analysed.")

    out = pd.DataFrame(rows)
    out.to_csv(outdir / "summary.csv", index=False)

    log("")
    log(f"Written {outdir}/summary.csv  ({len(out)} cultures)")
    log(f"  informative (presente escluso alla latitudine nota): "
        f"{int(out['informative'].sum())}")
    log(f"  senza taglio ammissibile (nessun p_sopra > p_sotto): "
        f"{int(out['fitted_h_min_deg'].isna().sum())}")
    log(f"  orizzonte pratico stimato: mediana "
        f"{np.nanmedian(out['fitted_h_min_deg']):.1f}°, "
        f"intervallo [{np.nanmin(out['fitted_h_min_deg']):.1f}, "
        f"{np.nanmax(out['fitted_h_min_deg']):.1f}]°")
    # h_min pinned at its bound would mean the degeneracy is still being
    # absorbed there rather than the horizon being measured.
    n_rail = int((out["fitted_h_min_deg"] >= args.hmin_max - 1e-9).sum())
    if n_rail:
        log(f"  ATTENZIONE: {n_rail} culture con h_min al limite "
            f"({args.hmin_max:.0f}°): latitudine e orizzonte restano degeneri, "
            f"usare phi_minus_hmin")


if __name__ == "__main__":
    main()
