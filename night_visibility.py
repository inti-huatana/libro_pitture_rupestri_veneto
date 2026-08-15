#!/usr/bin/env python3
"""
night_visibility.py

How much of the year the sky is dark enough to use, by latitude and epoch.

The usual statement is that astronomical night ends above 48.6 degrees of
latitude, because there the midnight Sun no longer reaches 18 degrees below the
horizon at the solstice. The arithmetic is right -- the Sun's altitude at lower
culmination is phi + delta - 90, and setting that to -18 with delta = 23.44
gives 48.56 -- but the criterion is not. Eighteen degrees is a photometric
convention, the depression beyond which twilight stops contaminating a
measurement of the sky background. It has nothing to do with an eye.

A bright star is seen long before that. The quantity that says when is the
arcus visionis, the solar depression at which a star of a given magnitude
becomes visible, and it is already the model this project uses for heliacal
risings: AV = a + b*V, with a = 10.5 degrees and b = 1.4 degrees per magnitude.
So the threshold is not chosen here, it is inherited, and it is a function of
the star rather than a constant of the calculation.

Reading the limit off that model instead of off the convention changes the
answer qualitatively. With AV(0) = 10.5 and AV(4) = 16.1 degrees the limiting
latitude is 56.1 for a first-magnitude star and 50.5 for a fourth-magnitude
one. Between those two latitudes there is a season in midsummer during which
the bright stars are up and visible but the faint members of their
constellations are not: the figures come apart, leaving isolated points with
nothing joining them. That band -- roughly Brittany to central Sweden -- is
where a constellation is a summer casualty rather than a permanent possession,
and it is invisible to any calculation that uses one threshold for all stars.

Both limits move with the obliquity, which swings between 22.05 and 24.50
degrees over 41 kyr, carrying each of them through about 2.4 degrees of
latitude, some 270 km.

Three things are computed.

    dark hours     For a grid of epochs and latitudes, and one magnitude class
                   at a time, the hours per year during which the Sun is more
                   than AV below the horizon, the number of nights on which
                   that happens at all, and the longest unbroken run of days on
                   which it never does. Sky only: no star enters here.

    limits         The latitude at which the darkest night of the year first
                   fails, epoch by epoch, for each magnitude class. Closed
                   form, 90 - eps(t) - AV(V), and it is the ladder above.

    stars          Given a trajectory file and one latitude, the observable
                   time of each star: the part of the night on which it stands
                   above a minimum altitude while the Sun is far enough down
                   for that star's own magnitude. This is the intersection of
                   two arcs of hour angle and is taken in closed form, not by
                   sampling the night.

The year is walked in true solar longitude rather than in days, and each step
weighted by the time the Earth spends in it, first order in the eccentricity.
That matters here and not only for tidiness: dark hours pile up in winter, and
whether perihelion falls in winter or in summer reverses over the 21 kyr in
which it circles the year, so the annual total carries a Milankovitch beat that
a uniform day grid would erase.

Usage
-----
    python3 night_visibility.py --outdir night
    python3 night_visibility.py --trajectories BSGRID/trajectories.csv \
            --lat 45.5 --outdir night
"""

from __future__ import annotations

import argparse
import math
import warnings
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd

from star_table import read_star_table, normalise_epoch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import erfa
except ImportError as exc:                                    # pragma: no cover
    raise SystemExit("Missing dependency 'pyerfa'. Install with: "
                     "pip install pyerfa") from exc

TROPICAL_YEAR_D = 365.2422

# Orbit shape. The eccentricity is held at its present value: it varies between
# roughly 0.005 and 0.06 on 100 and 400 kyr periods, but it enters only through
# the weighting of the year, where it is a few per cent correction, whereas the
# longitude of perihelion enters through its phase and must be carried.
ECCENTRICITY = 0.016708
PERIHELION_LON_J2000_DEG = 282.937
PERIHELION_PRECESSION_PERIOD_KYR = 20.9


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# the Sun through the year
# --------------------------------------------------------------------------

def obliquity_series(epochs_kyr: np.ndarray) -> np.ndarray:
    """Mean obliquity of date, degrees, as the angle between the two poles."""
    out = np.empty(epochs_kyr.size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, t in enumerate(epochs_kyr):
            y = 1000.0 * float(t)
            veq = np.asarray(erfa.ltpequ(y), dtype=float)
            vecl = np.asarray(erfa.ltpecl(y), dtype=float)
            out[i] = math.degrees(
                math.acos(np.clip(float(np.dot(veq, vecl)), -1.0, 1.0)))
    return out


def year_weights(epoch_kyr: float, n_steps: int):
    """Sample the year in true solar longitude and weight each step by its time.

    Returns the longitudes, the days each step lasts, and the day of the year
    each falls on, counted from the vernal equinox. The Earth is fastest at
    perihelion, so equal steps of longitude are not equal steps of time; to
    first order in e the ratio is 1 - 2e cos(nu), with nu the true anomaly.
    """
    lam = (np.arange(n_steps) + 0.5) * (2.0 * math.pi / n_steps)
    lon_p = math.radians(
        (PERIHELION_LON_J2000_DEG
         - 360.0 * epoch_kyr / PERIHELION_PRECESSION_PERIOD_KYR) % 360.0)
    w = 1.0 - 2.0 * ECCENTRICITY * np.cos(lam - lon_p)
    days = TROPICAL_YEAR_D * w / w.sum()
    doy = np.concatenate(([0.0], np.cumsum(days)[:-1])) + 0.5 * days
    return lam, days, doy


def sun_position(lam: np.ndarray, eps_deg: float):
    """Right ascension and declination of the Sun at those true longitudes."""
    e = math.radians(eps_deg)
    dec = np.arcsin(math.sin(e) * np.sin(lam))
    ra = np.arctan2(math.cos(e) * np.sin(lam), np.cos(lam))
    return ra, dec


# --------------------------------------------------------------------------
# hour angles and arcs
# --------------------------------------------------------------------------

def half_arc(dec_rad, lat_deg: float, alt_deg: float) -> np.ndarray:
    """Half the hour angle during which a body of that declination is above alt.

    Returns 0 where it never reaches that altitude and pi where it never drops
    below it, which are the two cases the arccos cannot express and where the
    notion of a rising and setting body stops applying.
    """
    p = math.radians(lat_deg)
    d = np.asarray(dec_rad, dtype=float)
    num = math.sin(math.radians(alt_deg)) - math.sin(p) * np.sin(d)
    den = math.cos(p) * np.cos(d)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(np.abs(den) > 1e-12, num / den, np.inf)
    return np.where(c <= -1.0, math.pi, np.where(c >= 1.0, 0.0, np.arccos(np.clip(c, -1.0, 1.0))))


def arc_overlap(sep, a1, a2):
    """Measure of the intersection of two arcs on a circle.

    Each arc is given by its half-width; sep is the angular distance between
    their centres. Two arcs on a circle can meet on both sides, so the overlap
    is the sum of the two contributions, one at separation sep and one at
    2*pi - sep, each capped at twice the shorter half-width.
    """
    sep = np.abs(np.asarray(sep, dtype=float)) % (2.0 * math.pi)
    a1 = np.asarray(a1, dtype=float)
    a2 = np.asarray(a2, dtype=float)
    cap = 2.0 * np.minimum(a1, a2)

    def side(x):
        return np.clip(a1 + a2 - x, 0.0, cap)

    return side(sep) + side(2.0 * math.pi - sep)


# --------------------------------------------------------------------------
# the sky window, without stars
# --------------------------------------------------------------------------

def dark_year(eps_deg: float, epoch_kyr: float, lat_deg: float,
              av_deg: float, n_steps: int):
    """Dark hours per year, dark nights, and the longest run of white nights."""
    lam, days, _ = year_weights(epoch_kyr, n_steps)
    _, dec = sun_position(lam, eps_deg)

    # Half the hour angle during which the Sun sits above -AV: the bright part
    # of the day. pi means it never gets that low, 0 that it never gets that
    # high, which at these latitudes is the depth of the polar night.
    h_bright = half_arc(dec, lat_deg, -av_deg)
    frac_dark = 1.0 - h_bright / math.pi
    hours = float(np.sum(frac_dark * days) * 24.0)
    n_dark = float(np.sum(days[frac_dark > 0.0]))

    white = frac_dark <= 0.0
    run = 0.0
    if white.any():
        # The year is a circle, so a run may straddle its start; doubling the
        # sequence and taking the longest run no longer than one year handles
        # that without a special case.
        d2, w2 = np.concatenate([days, days]), np.concatenate([white, white])
        cur = 0.0
        for k in range(w2.size):
            cur = cur + d2[k] if w2[k] else 0.0
            run = max(run, min(cur, TROPICAL_YEAR_D))
    return hours, n_dark, run


# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Usable darkness by latitude, epoch and stellar magnitude.")
    p.add_argument("--tmin", type=float, default=-100.0, help="kyr from year 0")
    p.add_argument("--tmax", type=float, default=0.0)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--lat-min", type=float, default=0.0)
    p.add_argument("--lat-max", type=float, default=75.0)
    p.add_argument("--lat-step", type=float, default=1.0)
    p.add_argument("--mags", type=float, nargs="+", default=[0.0, 2.0, 4.0],
                   help="magnitude classes the maps are drawn for")
    p.add_argument("--av-a", type=float, default=10.5,
                   help="arcus visionis intercept, deg (same as the grid)")
    p.add_argument("--av-b", type=float, default=1.4,
                   help="arcus visionis slope, deg per magnitude")
    p.add_argument("--steps", type=int, default=180,
                   help="samples of true solar longitude per year")
    p.add_argument("--trajectories", default=None,
                   help="optional: adds the per-star part")
    p.add_argument("--lat", type=float, default=45.5,
                   help="latitude for the per-star part and the year profile")
    p.add_argument("--h-min", type=float, default=5.0,
                   help="altitude a star must clear to count as seen, deg")
    p.add_argument("--vmax", type=float, default=4.0,
                   help="faintest star kept in the per-star part")
    p.add_argument("--outdir", default="night")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tmin, tmax = min(args.tmin, args.tmax), max(args.tmin, args.tmax)
    epochs = np.arange(tmin, tmax + 0.5 * args.dt, args.dt)
    lats = np.arange(args.lat_min, args.lat_max + 0.5 * args.lat_step,
                     args.lat_step)
    mags = np.array(sorted(args.mags), dtype=float)
    av = args.av_a + args.av_b * mags

    log(f"Epoche: {epochs.size} da {epochs[0]:+.1f} a {epochs[-1]:+.1f} kyr")
    log(f"Latitudini: {lats.size} da {lats[0]:.1f} a {lats[-1]:.1f}")
    log(f"Classi di magnitudine e arcus visionis:")
    for v, a in zip(mags, av):
        log(f"    V = {v:4.1f}   AV = {a:5.2f}°")

    log("Obliquita' di data ...")
    eps = obliquity_series(epochs)
    log(f"  da {eps.min():.3f}° a {eps.max():.3f}°")

    # ---- the ladder of limiting latitudes ---------------------------------
    # The darkest moment of the year is midnight at the solstice, where the Sun
    # reaches phi + eps - 90. Setting that to -AV gives the latitude at which
    # the last night of true darkness is lost, and it needs no search.
    rows = []
    for i, t in enumerate(epochs):
        for v, a in zip(mags, av):
            rows.append({"epoch_kyr": float(t), "obliquity_deg": float(eps[i]),
                         "Vmag": float(v), "arcus_visionis_deg": float(a),
                         "lat_limit_deg": float(90.0 - eps[i] - a)})
    lim = pd.DataFrame(rows)
    lim.to_csv(outdir / "latitude_limits.csv", index=False,
               float_format="%.4f")

    log("")
    log("  Latitudine oltre la quale si perde la notte utile al solstizio:")
    for v, a in zip(mags, av):
        s = lim[lim["Vmag"] == v]["lat_limit_deg"]
        log(f"    V = {v:4.1f}  ->  {s.mean():5.2f}°  "
            f"(da {s.min():5.2f}° a {s.max():5.2f}° con l'obliquita')")
    if mags.size >= 2:
        lo = lim[lim["Vmag"] == mags[-1]]["lat_limit_deg"].mean()
        hi = lim[lim["Vmag"] == mags[0]]["lat_limit_deg"].mean()
        log(f"    fascia in cui d'estate restano solo le stelle brillanti: "
            f"{lo:5.2f}° .. {hi:5.2f}°")

    # ---- the maps ----------------------------------------------------------
    log("")
    log("Mappe del buio utile ...")
    hours = np.empty((mags.size, epochs.size, lats.size))
    nights = np.empty_like(hours)
    white = np.empty_like(hours)
    for m, a in enumerate(av):
        for i, t in enumerate(epochs):
            for j, phi in enumerate(lats):
                hours[m, i, j], nights[m, i, j], white[m, i, j] = dark_year(
                    eps[i], float(t), float(phi), float(a), args.steps)
        log(f"  V = {mags[m]:4.1f} fatta")

    rec = []
    for m, v in enumerate(mags):
        for i, t in enumerate(epochs):
            for j, phi in enumerate(lats):
                rec.append({"epoch_kyr": float(t), "lat_deg": float(phi),
                            "Vmag": float(v),
                            "arcus_visionis_deg": float(av[m]),
                            "dark_hours_per_year": hours[m, i, j],
                            "dark_nights_per_year": nights[m, i, j],
                            "white_season_days": white[m, i, j]})
    pd.DataFrame(rec).to_csv(outdir / "dark_hours.csv", index=False,
                             float_format="%.3f")

    now = int(np.argmin(np.abs(epochs - epochs.max())))
    log("")
    log(f"  A {args.lat:.1f}° di latitudine, epoca {epochs[now]:+.1f} kyr:")
    jl = int(np.argmin(np.abs(lats - args.lat)))
    for m, v in enumerate(mags):
        log(f"    V = {v:4.1f}: {hours[m, now, jl]:7.1f} h/anno di cielo "
            f"abbastanza buio, {nights[m, now, jl]:5.1f} notti, "
            f"stagione bianca {white[m, now, jl]:5.1f} giorni")

    ext = [lats[0], lats[-1], epochs[0], epochs[-1]]
    fig, axes = plt.subplots(2, mags.size, figsize=(4.6 * mags.size, 8.4),
                             squeeze=False)
    for m, v in enumerate(mags):
        a1, a2 = axes[0][m], axes[1][m]
        im = a1.imshow(hours[m], origin="lower", aspect="auto", extent=ext,
                       cmap="magma")
        fig.colorbar(im, ax=a1, label="ore/anno")
        a1.set_title(f"buio utile per V = {v:.1f}  (AV {av[m]:.1f}°)")
        a1.set_ylabel("epoca [kyr]")

        im = a2.imshow(white[m], origin="lower", aspect="auto", extent=ext,
                       cmap="viridis")
        fig.colorbar(im, ax=a2, label="giorni")
        a2.set_title("stagione bianca")
        a2.set_xlabel("latitudine [°]")
        if m == 0:
            a2.set_ylabel("epoca [kyr]")
        for ax in (a1, a2):
            ax.plot(lim[lim["Vmag"] == v]["lat_limit_deg"], epochs,
                    color="w", lw=1.0, ls="--")
    fig.suptitle("Quanta notte resta, per latitudine ed epoca\n"
                 "tratteggio: latitudine oltre la quale al solstizio "
                 "il cielo non si fa mai abbastanza buio")
    fig.tight_layout()
    fig.savefig(outdir / "dark_hours.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for m, v in enumerate(mags):
        ax.plot(epochs, lim[lim["Vmag"] == v]["lat_limit_deg"], lw=1.4,
                label=f"V = {v:.1f}  (AV {av[m]:.1f}°)")
    ax.axhline(args.lat, color="k", lw=0.8, ls=":",
               label=f"latitudine {args.lat:.1f}°")
    ax.set_xlabel("epoca [kyr dall'anno 0]")
    ax.set_ylabel("latitudine limite [°]")
    ax.set_title("La scala delle latitudini limite, e come l'obliquita' la muove")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "latitude_limits.png", dpi=args.dpi,
                bbox_inches="tight")
    plt.close(fig)

    # ---- the year profile at one latitude ---------------------------------
    lam, days, doy = year_weights(float(epochs[now]), args.steps)
    _, sdec = sun_position(lam, float(eps[now]))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for m, v in enumerate(mags):
        hb = half_arc(sdec, args.lat, -float(av[m]))
        ax.plot(doy, 24.0 * (1.0 - hb / math.pi), lw=1.4,
                label=f"V = {v:.1f}")
    ax.set_xlabel("giorno dell'anno dall'equinozio di primavera")
    ax.set_ylabel("ore di cielo abbastanza buio")
    ax.set_title(f"Andamento annuo a {args.lat:.1f}°, epoca "
                 f"{epochs[now]:+.1f} kyr")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "annual_profile.png", dpi=args.dpi,
                bbox_inches="tight")
    plt.close(fig)

    log("")
    log(f"Scritti {outdir}/dark_hours.csv, latitude_limits.csv, "
        f"dark_hours.png, latitude_limits.png, annual_profile.png")

    if not args.trajectories:
        log("")
        log("Parte per stella non fatta: passare --trajectories per averla.")
        return

    # ---- the per-star part -------------------------------------------------
    log("")
    log(f"Leggo {args.trajectories} ...")
    traj = read_star_table(args.trajectories,
                           ["epoch_kyr_from_year0", "HIP", "ra_deg", "dec_deg",
                            "Vmag", "NAME", "Bayer"])
    traj = normalise_epoch(traj).drop_duplicates(["HIP", "epoch"])
    traj = traj[traj["Vmag"] <= args.vmax]
    traj = traj[(traj["epoch"] >= tmin - 1e-6) & (traj["epoch"] <= tmax + 1e-6)]
    if traj.empty:
        raise SystemExit("Nessuna riga di traiettoria nell'intervallo chiesto.")

    lab = traj.groupby("HIP")[["NAME", "Bayer"]].first()
    labels = {}
    for hip, r in lab.iterrows():
        nm, by = str(r["NAME"]).strip(), str(r["Bayer"]).strip()
        labels[int(hip)] = (nm if nm and nm != "nan"
                            else by if by and by != "nan" else f"HIP {int(hip)}")

    ra_w = traj.pivot(index="epoch", columns="HIP", values="ra_deg").sort_index()
    de_w = traj.pivot(index="epoch", columns="HIP", values="dec_deg").sort_index()
    mg_w = traj.pivot(index="epoch", columns="HIP", values="Vmag").sort_index()
    st_ep = ra_w.index.to_numpy(dtype=float)
    hips = ra_w.columns.to_numpy()
    log(f"  {hips.size} stelle V<{args.vmax}, {st_ep.size} epoche")
    log(f"  latitudine {args.lat:.1f}°, altezza minima {args.h_min:.1f}°")

    eps_st = obliquity_series(st_ep)
    n_ep, n_st = st_ep.size, hips.size
    obs_h = np.zeros((n_ep, n_st))          # observable hours per year
    obs_n = np.zeros((n_ep, n_st))          # nights with any observable time
    up_h = np.zeros((n_ep, n_st))           # hours above h_min, dark or not

    ra_a = np.radians(ra_w.to_numpy(dtype=float))
    de_a = np.radians(de_w.to_numpy(dtype=float))
    mg_a = mg_w.to_numpy(dtype=float)

    for i in range(n_ep):
        lam, days, _ = year_weights(float(st_ep[i]), args.steps)
        sra, sdec = sun_position(lam, float(eps_st[i]))
        av_i = args.av_a + args.av_b * mg_a[i]                  # (n_st,)

        # Bright arc of the Sun, one half-width per day and per magnitude.
        # h_bright[d, s]: the Sun is above -AV(s) for |H| <= h_bright.
        h_bright = np.empty((lam.size, n_st))
        for s in range(n_st):
            h_bright[:, s] = half_arc(sdec, args.lat, -float(av_i[s]))

        # Arc during which the star clears h_min; it does not depend on the day.
        h_star = half_arc(de_a[i], args.lat, args.h_min)         # (n_st,)

        # The star's arc is centred on its own meridian passage, the Sun's on
        # local noon, and the two are separated by the difference in right
        # ascension. Observable time is the star's arc minus what the Sun's
        # bright arc eats out of it.
        sep = sra[:, None] - ra_a[i][None, :]
        over = arc_overlap(sep, np.broadcast_to(h_star, h_bright.shape),
                           h_bright)
        arc = np.maximum(2.0 * h_star[None, :] - over, 0.0)

        frac = arc / (2.0 * math.pi)
        obs_h[i] = np.sum(frac * days[:, None], axis=0) * 24.0
        obs_n[i] = np.sum(days[:, None] * (frac > 1e-9), axis=0)
        up_h[i] = np.sum((2.0 * h_star[None, :] / (2.0 * math.pi))
                         * days[:, None], axis=0) * 24.0
        if (i + 1) % 25 == 0 or i == n_ep - 1:
            log(f"  {i + 1}/{n_ep} epoche")

    rows = []
    for i in range(n_ep):
        for s in range(n_st):
            rows.append({"epoch_kyr": float(st_ep[i]), "lat_deg": args.lat,
                         "HIP": int(hips[s]), "star": labels[int(hips[s])],
                         "Vmag": float(mg_a[i, s]),
                         "arcus_visionis_deg": float(args.av_a
                                                     + args.av_b * mg_a[i, s]),
                         "hours_above_horizon": float(up_h[i, s]),
                         "hours_observable": float(obs_h[i, s]),
                         "nights_observable": float(obs_n[i, s]),
                         "lost_to_twilight_frac": float(
                             1.0 - obs_h[i, s] / up_h[i, s]) if up_h[i, s] > 0
                         else np.nan})
    sv = pd.DataFrame(rows)
    sv.to_csv(outdir / "star_visibility.csv", index=False, float_format="%.4f")

    j = int(np.argmin(np.abs(st_ep - st_ep.max())))
    cur = sv[np.isclose(sv["epoch_kyr"], st_ep[j])].sort_values(
        "hours_observable", ascending=False)
    log("")
    log(f"  Epoca {st_ep[j]:+.1f} kyr, {args.lat:.1f}°: le piu' e le meno "
        f"osservabili")
    for _, r in pd.concat([cur.head(8), cur.tail(5)]).iterrows():
        log(f"    {r['star']:18s} V {r['Vmag']:4.1f}  "
            f"sopra l'orizzonte {r['hours_above_horizon']:6.0f} h/anno, "
            f"osservabili {r['hours_observable']:6.0f} h "
            f"({100 * (1 - r['lost_to_twilight_frac']):4.1f}%), "
            f"{r['nights_observable']:5.0f} notti")

    tot = sv.groupby("epoch_kyr")["hours_observable"].sum()
    log("")
    log(f"  Ore osservabili sommate su tutte le stelle: da {tot.min():,.0f} "
        f"a {tot.max():,.0f} h/anno")

    order = np.argsort(-obs_h[j])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 9))
    im = a1.imshow(obs_h[:, order].T, origin="lower", aspect="auto",
                   extent=[epochs[0], epochs[-1], 0, n_st], cmap="magma")
    fig.colorbar(im, ax=a1, label="ore osservabili/anno")
    a1.set_ylabel("stelle, ordinate per osservabilita' attuale")
    a1.set_title(f"Tempo osservabile per stella a {args.lat:.1f}°, "
                 f"h > {args.h_min:.0f}°, soglia di crepuscolo dalla "
                 f"magnitudine di ciascuna")

    a2.plot(st_ep, tot.reindex(st_ep).to_numpy(), lw=1.4, color="navy")
    a2.set_xlabel("epoca [kyr dall'anno 0]")
    a2.set_ylabel("ore osservabili, somma su tutte le stelle")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "star_visibility.png", dpi=args.dpi,
                bbox_inches="tight")
    plt.close(fig)

    log("")
    log(f"Scritti {outdir}/star_visibility.csv, star_visibility.png")


if __name__ == "__main__":
    main()
