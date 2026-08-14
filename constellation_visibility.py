#!/usr/bin/env python3
"""
constellation_visibility.py

Constrain the observing latitude and the epoch of a sky culture from the
altitudes at which the stars it names culminate.

Supersedes the empty-zone approach of constellation_age.py, which inferred from
an absence -- no constellations near the south celestial pole -- and so could not
work for a southern culture at all. This one infers from a presence: a star
belongs to the culture's canon, so somebody saw it, so it rose. That is physical
evidence rather than a fit to a hole, and it is symmetric in the hemispheres.

The estimator
-------------
A star of declination d culminates at altitude h = 90 - |phi - d| for an observer
at latitude phi, so at a candidate (phi, epoch) every star of the bright-star
catalogue has a known culmination altitude and is either in the canon or not.
If the culture observed from there, naming should fall off towards the horizon:
atmospheric extinction is about 0.8 mag at 15 degrees of altitude, 2 mag at 5 and
4 mag at 2, and a star that only ever skims the horizon is useless for building a
figure.

That fall-off is fitted as a logistic regression of canon membership on

    v = max(h, 0)          altitude above the horizon, zero when the star
                           never rises, since below the horizon there are no
                           degrees of invisibility to distinguish

    V                      apparent magnitude at that epoch, carried as a
                           control: brighter stars are likelier to be named for
                           reasons that have nothing to do with altitude, and
                           without it that preference would leak into the slope

    logit p(in canon) = a + b*v + c*V

and the quantity of interest is b, the strength of the altitude effect. A real
horizon gives b > 0; b = 0 says the canon carries no information about where it
was observed from. The likelihood ratio against b = 0 has one degree of freedom
and, b being an ordinary regression coefficient, is properly calibrated.

Why there is no cutoff parameter here
-------------------------------------
An earlier version modelled the fall-off as a step: one naming rate above a
fitted altitude h_min and another below. That model is pathological. Raising the
threshold sweeps non-canon stars into the lower group and costs nothing until it
catches the first canon star, so the likelihood increases monotonically and the
maximum sits exactly at min(h) over the canon. The fitted "practical horizon" was
therefore an order statistic wearing the clothes of a parameter, and it pinned
itself to whatever bound the grid imposed for every culture whose canon does not
reach the horizon in the first place -- which is most northern ones, whose stars
are circumpolar. Holding the latitude fixed did not help, because the defect was
in the step, not in the latitude. A logistic slope has an interior maximum and
does not have this failure mode.

Perfect separation, where no canon star lies low and the slope would run to
infinity, is held finite by a small ridge penalty on the coefficients.

What to look for
----------------
Modern reconstructions are built by scholars who know where a culture lived and
would not have included stars invisible from there today, so a canon consistent
with its own latitude at the present epoch is close to circular. The informative
case is the opposite one, flagged as `informative`:

    the present epoch is excluded at the known latitude, while some past epoch
    is not

which no reconstruction can have manufactured. Expect few: for most cultures the
constraint is wide and the present perfectly admissible, and that is the correct
answer rather than a failure. Sets covering the whole sky by construction, the
`modern*` ones, should return b near zero, and are the control.

Usage
-----
    python3 constellation_visibility.py \
        --trajectories BSGRID/trajectories.csv \
        --skycultures skycultures_csv \
        --outdir constellation_visibility_out

Output
------
    summary.csv              one row per culture: the maximum, the fitted slope
                             and its error, the likelihood ratio, the 95% ranges
                             and the verdict on the present epoch

    surfaces/<culture>.csv   the profile surface in long form, one row per
                             (epoch, latitude), with the log-likelihood, its
                             distance from the maximum, whether the cell is
                             inside the 95% region, and the fitted coefficients

    surfaces/<culture>.pdf   the same surface drawn, with the known latitude and
                             the present epoch marked
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
    "chinese_chenzhuo": 34.3,
    "chinese_contemporary": 34.3,
    "chinese_medieval": 34.3,
    "chinese_song_dynasty": 34.3,
    "dakota": 44.5,
    "egyptian": 25.7,                    # Thebes
    "egyptian_dendera": 26.1,            # Dendera
    "greek_almagest": 31.2,              # Alexandria
    "greek_dante": 43.8,                 # Florence
    "greek_farnese": 31.2,
    "greek_leidenAratea": 31.2,
    "hawaiian_starlines": 20.8,
    "indian": 25.0,
    "indian_nakshatras": 25.0,
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
    "norse_edda": 64.0,                  # Iceland
    "northern_andes": 4.6,
    "ojibwe": 47.0,
    "romanian": 45.9,
    "ruanui_sky_tahiti_and_society_islands": -17.6,
    "russian_siberian": 58.0,
    "sami": 68.4,
    "samoan": -13.8,
    "sardinian": 40.1,
    "seri": 29.0,                        # Sonora, Mexico
    "siberian": 58.0,
    "tibetan": 29.7,                     # Lhasa
    "tikuna": -4.0,                      # upper Amazon
    "tongan": -21.1,
    "tukano": -0.5,                      # NW Amazon
    "tupi": -10.0,
    "vanuatu_netwar": -19.5,             # Tanna
    "western": 37.0,
    "western_hlad": 37.0,
    "western_rey": 37.0,
}

# Sets that cover the whole sky by construction and therefore must show no
# altitude effect; they are the control on the whole procedure.
ALLSKY_CONTROLS = ("modern", "modern_chinese", "modern_hlad", "modern_iau",
                   "modern_journey_to_the_west", "modern_rey", "modern_st")


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------------
def logistic_loglik(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    eta = X @ beta
    # log(1 + exp(eta)) via logaddexp, stable for large |eta|.
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))


def fit_logistic(X: np.ndarray, y: np.ndarray, ridge: float,
                 max_iter: int = 40, tol: float = 1e-9):
    """Ridge-penalised logistic regression by Newton iteration.

    The ridge term keeps the fit finite under perfect separation, which happens
    whenever no canon star lies low enough to contradict the altitude effect and
    the unpenalised slope would run away to infinity. It is small enough not to
    matter otherwise, and identical across cells, so it does not distort the
    comparison between them.
    """
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(max_iter):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -500.0, 500.0)))
        w = np.clip(mu * (1.0 - mu), 1e-10, None)
        grad = X.T @ (y - mu) - ridge * beta
        H = (X.T * w) @ X + ridge * np.eye(p)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta, logistic_loglik(X, y, beta), H


def cell_fit(v: np.ndarray, vmag: np.ndarray, y: np.ndarray, ridge: float):
    """Fit the altitude effect at one (latitude, epoch). Returns slope, error,
    intercept, magnitude coefficient and the maximised log-likelihood."""
    X = np.column_stack((np.ones_like(v), v, vmag))
    beta, ll, H = fit_logistic(X, y, ridge)
    try:
        se = float(np.sqrt(np.linalg.inv(H)[1, 1]))
    except np.linalg.LinAlgError:
        se = np.nan
    return float(beta[1]), se, float(beta[0]), float(beta[2]), ll


def argmax_positive_slope(ll: np.ndarray, slope: np.ndarray):
    """Best cell among those with a positive altitude effect.

    A horizon can only suppress naming, so b <= 0 is not a physical solution but
    the mirror of one: h = 90 - |phi - d| is symmetric under reflection, and
    placing a northern canon far enough south turns the fall-off upside down,
    giving a strong fit to nothing. Those mirrors are what used to drive the
    maximum onto the southern edge of the grid.
    """
    masked = np.where(slope > 0.0, ll, -np.inf)
    if not np.isfinite(masked).any():
        return None
    return np.unravel_index(int(np.argmax(masked)), ll.shape)


def permutation_max_lr(dec, vmag, y, lat_grid, ridge, n_perm, rng,
                       lat_stride: int = 4, epoch_stride: int = 2):
    """Null distribution of the maximised likelihood ratio.

    The reported ratio is a maximum over the whole grid, so it is not a
    one-degree-of-freedom chi-square however regular the slope is: with several
    thousand cells the largest of them lands near 2*ln(n_cells) under pure noise.
    Shuffling canon membership and repeating the maximisation calibrates it
    directly. The grid is coarsened for the replicas, which understates the
    maximum slightly and so is conservative in the safe direction.
    """
    lat_c = lat_grid[::lat_stride]
    dec_c = dec[::epoch_stride]
    mag_c = vmag[::epoch_stride]
    i_ref = min(dec_c.shape[0] - 1, dec.shape[0] // 2)

    out = np.empty(n_perm)
    for r in range(n_perm):
        yp = rng.permutation(y)
        ll, slope, _, _, _ = surface(dec_c, mag_c, yp, lat_c, ridge)
        ll0 = null_fit(mag_c[i_ref], yp, ridge)
        best = argmax_positive_slope(ll, slope)
        out[r] = 0.0 if best is None else 2.0 * (ll[best] - ll0)
    return out


def null_fit(vmag: np.ndarray, y: np.ndarray, ridge: float) -> float:
    """Log-likelihood with the altitude effect removed, b = 0.

    It does not involve the altitude at all, so it is the same for every cell of
    the surface and is computed once per culture.
    """
    X = np.column_stack((np.ones_like(vmag), vmag))
    _, ll, _ = fit_logistic(X, y, ridge)
    return ll


def surface(dec: np.ndarray, vmag: np.ndarray, y: np.ndarray,
            lat_grid: np.ndarray, ridge: float):
    """Profile log-likelihood and fitted coefficients over (epoch, latitude)."""
    n_epoch = dec.shape[0]
    n_lat = lat_grid.size

    ll = np.full((n_epoch, n_lat), -np.inf)
    slope = np.full((n_epoch, n_lat), np.nan)
    slope_se = np.full((n_epoch, n_lat), np.nan)
    inter = np.full((n_epoch, n_lat), np.nan)
    cmag = np.full((n_epoch, n_lat), np.nan)

    for i in range(n_epoch):
        di = dec[i]
        vi = vmag[i]
        for j, phi in enumerate(lat_grid):
            v = np.maximum(90.0 - np.abs(phi - di), 0.0)
            b, se, a, c, value = cell_fit(v, vi, y, ridge)
            ll[i, j] = value
            slope[i, j] = b
            slope_se[i, j] = se
            inter[i, j] = a
            cmag[i, j] = c

    return ll, slope, slope_se, inter, cmag


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_surface_csv(path: Path, epochs, lat_grid, ll, slope, slope_se,
                      inter, cmag, ll_best: float, ll_null: float) -> int:
    n_epoch, n_lat = ll.shape
    flat = ll.ravel()
    df = pd.DataFrame({
        "epoch_kyr": np.repeat(epochs, n_lat),
        "lat_deg": np.tile(lat_grid, n_epoch),
        "loglik": flat,
        "delta_2loglik": 2.0 * (ll_best - flat),
        "inside_95": 2.0 * (ll_best - flat) <= CHI2_2DOF_95,
        "lr_slope": 2.0 * (flat - ll_null),
        "slope_per_deg": slope.ravel(),
        "slope_se": slope_se.ravel(),
        "intercept": inter.ravel(),
        "coef_vmag": cmag.ravel(),
    })
    df.to_csv(path, index=False, float_format="%.5f")
    return len(df)


def plot_profile(culture, epochs, ll, slope, lat_known, present_epoch, path):
    dev = 2.0 * (np.nanmax(ll) - ll)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(epochs, dev, lw=1.4)
    ax1.axhline(CHI2_1DOF_95, color="crimson", lw=1.0, ls="--",
                label="soglia 95% (1 g.d.l.)")
    ax1.axvline(present_epoch, color="k", lw=1.0, ls=":", label="presente")
    ax1.set_ylabel(r"$2\,\Delta\log L$ dal massimo")
    ax1.set_ylim(bottom=0)
    ax1.set_title(f"{culture} — latitudine fissata a {lat_known:+.1f}°")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, slope, lw=1.4, color="darkgreen")
    ax2.axhline(0.0, color="k", lw=0.8)
    ax2.axvline(present_epoch, color="k", lw=1.0, ls=":")
    ax2.set_xlabel("Epoca [kyr dall'anno 0]")
    ax2.set_ylabel("Pendenza logit per grado")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_culture(culture, epochs, lat_grid, ll, lat_known, present_epoch,
                 best, path):
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
    p.add_argument("--pin-latitude", action="store_true",
                   help="hold the latitude at its known value and fit the epoch "
                        "only; cultures without a known latitude are skipped")
    p.add_argument("--ridge", type=float, default=1e-3,
                   help="ridge penalty keeping the slope finite under perfect "
                        "separation")
    p.add_argument("--present-kyr", type=float, default=2.0,
                   help="epoch taken as the present, kyr from year 0")
    p.add_argument("--min-stars", type=int, default=10,
                   help="skip cultures with fewer catalogued stars")
    p.add_argument("--permutations", type=int, default=0,
                   help="replicas calibrating the maximised likelihood ratio "
                        "against the search over the grid; 0 skips, and the "
                        "all-sky controls then serve as the empirical null")
    p.add_argument("--seed", type=int, default=20260814)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    outdir = Path(args.outdir)
    (outdir / "surfaces").mkdir(parents=True, exist_ok=True)

    latitudes = dict(DEFAULT_LATITUDES)
    if args.latitudes:
        ext = pd.read_csv(args.latitudes)
        latitudes.update(dict(zip(ext["culture"], ext["latitude"].astype(float))))
        log(f"Latitude table overridden for {len(ext)} cultures")

    log(f"Reading {args.trajectories} ...")
    traj = pd.read_csv(args.trajectories,
                       usecols=["epoch_kyr_from_year0", "HIP", "dec_deg", "Vmag"])
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

    dec_wide = traj.pivot(index="epoch", columns="HIP", values="dec_deg").sort_index()
    mag_wide = traj.pivot(index="epoch", columns="HIP", values="Vmag").sort_index()
    epochs = dec_wide.index.to_numpy(dtype=float)
    all_hips = dec_wide.columns.to_numpy()
    dec_all = dec_wide.to_numpy(dtype=float)
    mag_all = mag_wide.to_numpy(dtype=float)
    log(f"Epochs: {epochs.size} from {epochs.min():+.1f} to {epochs.max():+.1f} kyr")

    lat_grid_full = np.arange(args.lat_min, args.lat_max + 1e-9, args.lat_step)
    log(f"Latitudes: {lat_grid_full.size} from {lat_grid_full[0]:+.0f} "
        f"to {lat_grid_full[-1]:+.0f}")
    if args.pin_latitude:
        log("Latitude pinned to the known value: fitting the epoch only")

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
        y = np.isin(all_hips, hips).astype(float)
        n_present = int(y.sum())
        if n_present < args.min_stars:
            log(f"  {culture:28s} SKIP: {n_present} stelle nel catalogo")
            continue

        lat_known = latitudes.get(culture)
        if args.pin_latitude:
            if lat_known is None:
                log(f"  {culture:28s} SKIP: latitudine non nota, richiesta "
                    f"da --pin-latitude")
                continue
            lat_grid = np.array([lat_known])
        else:
            lat_grid = lat_grid_full

        ll_null = null_fit(mag_all[i_present], y, args.ridge)
        ll, slope, slope_se, inter, cmag = surface(
            dec_all, mag_all, y, lat_grid, args.ridge)

        pos = argmax_positive_slope(ll, slope)
        if pos is None:
            log(f"  {culture:28s} SKIP: nessuna cella con effetto orizzonte "
                f"positivo")
            continue
        bi, bj = pos
        best = {"epoch": float(epochs[bi]), "lat": float(lat_grid[bj])}
        lr_slope = 2.0 * (ll[bi, bj] - ll_null)

        dev = 2.0 * (ll[bi, bj] - ll)
        inside = dev <= CHI2_2DOF_95
        lat_lo = float(lat_grid[inside.any(axis=0)].min())
        lat_hi = float(lat_grid[inside.any(axis=0)].max())
        ep_lo = float(epochs[inside.any(axis=1)].min())
        ep_hi = float(epochs[inside.any(axis=1)].max())

        write_surface_csv(outdir / "surfaces" / f"{culture}.csv",
                          epochs, lat_grid, ll, slope, slope_se, inter, cmag,
                          float(ll[bi, bj]), ll_null)
        if lat_grid.size > 1:
            plot_culture(culture, epochs, lat_grid, ll, lat_known,
                         present_epoch, best,
                         outdir / "surfaces" / f"{culture}.pdf")
        else:
            plot_profile(culture, epochs, ll[:, 0], slope[:, 0], lat_known,
                         present_epoch, outdir / "surfaces" / f"{culture}.pdf")

        b = float(slope[bi, bj])
        a = float(inter[bi, bj])
        c = float(cmag[bi, bj])
        vbar = float(np.nanmean(mag_all[bi]))

        def logistic(x):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))

        p_hor = float(logistic(a + c * vbar))
        p_45 = float(logistic(a + 45.0 * b + c * vbar))
        # A fit predicting a negligible naming probability at every altitude has
        # separated: no canon star lies low enough to contradict the effect, the
        # coefficient runs away and only the ridge holds it. It happens on small
        # canons that sit entirely high in the sky, and the very large slope it
        # produces is an artefact rather than a strong effect.
        separated = bool(p_45 < 0.01 or p_hor < 1e-4 or abs(b) > 0.15)

        row = {
            "culture": culture,
            "is_allsky_control": culture in ALLSKY_CONTROLS,
            "separated_fit": separated,
            "n_stars_culture": len(hips),
            "n_stars_present": n_present,
            "coverage": n_present / len(hips),
            "best_epoch_kyr": best["epoch"],
            "best_epoch_year": best["epoch"] * 1000.0,
            "best_latitude_deg": best["lat"],
            "slope_per_deg": b,
            "slope_se": float(slope_se[bi, bj]),
            "slope_z": b / slope_se[bi, bj] if slope_se[bi, bj] > 0 else np.nan,
            "lr_slope": lr_slope,
            # Effect size in the units the reader cares about: how much likelier
            # a star of average brightness is to be named high up than at the
            # horizon, at the best-fitting latitude and epoch.
            "p_at_horizon": p_hor,
            "p_at_45deg": p_45,
            "lat_lo95_deg": lat_lo, "lat_hi95_deg": lat_hi,
            "epoch_lo95_kyr": ep_lo, "epoch_hi95_kyr": ep_hi,
            "lat_known_deg": lat_known if lat_known is not None else np.nan,
        }

        if lat_known is None:
            row.update(present_excluded=np.nan, best_epoch_at_known_kyr=np.nan,
                       lr_present=np.nan, informative=False,
                       slope_at_known=np.nan, lr_slope_at_known=np.nan,
                       notes="latitudine non nota: solo superficie 2D")
        else:
            jk = int(np.argmin(np.abs(lat_grid - lat_known)))
            prof = ll[:, jk]
            # The one test with no search behind it: latitude fixed by
            # ethnography, epoch fixed at the present, so the ratio really does
            # have one degree of freedom and the 3.84 threshold applies.
            lr_known = 2.0 * (prof[i_present] - ll_null)
            ib = int(np.argmax(prof))
            lr_present = 2.0 * (prof[ib] - prof[i_present])
            excluded = bool(lr_present > CHI2_1DOF_95)
            row.update(best_epoch_at_known_kyr=float(epochs[ib]),
                       slope_at_known=float(slope[i_present, jk]),
                       lr_slope_at_known=float(lr_known),
                       lr_present=float(lr_present),
                       present_excluded=excluded,
                       # Only a culture that shows a real horizon effect where it
                       # actually lived can say anything about when it looked.
                       informative=bool(excluded
                                        and slope[i_present, jk] > 0.0
                                        and lr_known > CHI2_1DOF_95
                                        and not separated),
                       notes="")

        if args.permutations > 0:
            null_lr = permutation_max_lr(dec_all, mag_all, y, lat_grid,
                                         args.ridge, args.permutations, rng)
            row["p_lr_permutation"] = float((null_lr >= lr_slope).mean())
            row["lr_null_median"] = float(np.median(null_lr))
        else:
            row["p_lr_permutation"] = np.nan
            row["lr_null_median"] = np.nan

        rows.append(row)

        lat_txt = (f"lat nota {lat_known:+5.1f}°" if lat_known is not None
                   else "lat ignota    ")
        verdict = ("INFORMATIVA" if row["informative"]
                   else ("presente ammesso" if lat_known is not None else "--"))
        log(f"  {culture:28s} {n_present:4d} st | max ({best['epoch']:+6.1f} kyr, "
            f"{best['lat']:+5.1f}°) | b {b:+7.4f}/° (z {row['slope_z']:5.1f}) | "
            f"p {row['p_at_horizon']:.3f}->{row['p_at_45deg']:.3f} | "
            f"LR {lr_slope:7.1f} | {lat_txt} | {verdict}")

    if not rows:
        raise SystemExit("No culture could be analysed.")

    out = pd.DataFrame(rows)
    out.to_csv(outdir / "summary.csv", index=False)

    n_cells = epochs.size * lat_grid_full.size
    look_elsewhere = 2.0 * np.log(n_cells)

    log("")
    log(f"Written {outdir}/summary.csv  ({len(out)} cultures)")
    log(f"  informative: {int(out['informative'].sum())}")
    log(f"  fit separati (pendenza non attendibile): "
        f"{int(out['separated_fit'].sum())}")
    log(f"  pendenza stimata: mediana {np.nanmedian(out['slope_per_deg']):+.4f}/°, "
        f"quartili [{np.nanpercentile(out['slope_per_deg'], 25):+.4f}, "
        f"{np.nanpercentile(out['slope_per_deg'], 75):+.4f}]")

    log("")
    log(f"  lr_slope e' un massimo su {n_cells:,} celle, non un chi-quadro a un "
        f"grado di liberta':")
    log(f"    sotto rumore puro il massimo vale circa 2*ln(n) = "
        f"{look_elsewhere:.1f}, non 3.84")
    log(f"    il test calibrato e' lr_slope_at_known, a latitudine nota ed "
        f"epoca presente, senza ricerca dietro")

    ctrl = out[out["is_allsky_control"]]
    if not ctrl.empty:
        log("")
        log("  Controllo, insiemi che coprono tutto il cielo "
            "(danno la soglia empirica del rumore):")
        for _, r in ctrl.iterrows():
            log(f"    {r['culture']:28s} b {r['slope_per_deg']:+.4f}/° "
                f"z {r['slope_z']:5.1f}  LR {r['lr_slope']:7.1f}")
        log(f"    -> LR mediano dei controlli {ctrl['lr_slope'].median():.1f}, "
            f"massimo {ctrl['lr_slope'].max():.1f}; sotto questo valore "
            f"nessun risultato e' distinguibile dal rumore")


if __name__ == "__main__":
    main()
