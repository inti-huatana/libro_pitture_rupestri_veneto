#!/usr/bin/env python3
"""
earth_orientation.py

The orientation of the Earth's axis and orbit over the simulated span, with no
stars involved.

bright_star_grid.py computes the obliquity at every epoch and then discards it,
yet three separate questions depend on it and nothing else. This program keeps
it, together with the quantities that follow from it, as a table the other
programs read.

What comes out
--------------
    obliquity           the tilt of the axis, from ERFA's long-term model
                        (Vondrak et al. 2011). It swings between roughly 22.0
                        and 24.5 degrees with a period near 41 kyr, and every
                        line below follows from it.

    solstice azimuth    where the Sun rises at midsummer, per latitude. Because
                        the obliquity moves, so does this: at the latitude of the
                        Veneto the swing is close to four degrees over the cycle,
                        far more than the tolerance with which an alignment is
                        ever claimed. Any statement that a site faces the
                        solstice has to name the epoch it means.

    standstill azimuth  the same for the Moon at its major and minor standstills,
                        whose declinations are the obliquity plus and minus the
                        lunar orbital inclination of 5.145 degrees. This needs no
                        ephemeris: the standstill limits are geometry, not a
                        position on a given night, and the inclination is
                        effectively constant.

    galactic pole       the direction of the galactic pole in the equatorial
                        frame of date, from which the tilt of the Milky Way
                        against the horizon follows at any latitude.

    season lengths      the intervals between equinoxes and solstices. They are
                        set by where perihelion falls, which circles the year in
                        about 21 kyr, so they are not the fixed 92.8 / 93.6 /
                        89.8 / 89.0 days of the present epoch. stellar_calendar.py
                        numbers its days on that assumption and drifts by a few
                        days at remote epochs; this table is what corrects it.

Usage
-----
    python3 earth_orientation.py --Tmin -200 --Tmax 2 --dt 0.1 --outdir earth
"""

from __future__ import annotations

import argparse
import math
import warnings
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import erfa
except ImportError as exc:
    raise SystemExit("Missing dependency 'pyerfa'. Install with: pip install pyerfa") from exc

# Inclination of the lunar orbit to the ecliptic, degrees. Effectively constant:
# it oscillates by some nine arcminutes with an 18.6-year period and has no
# secular drift worth carrying over these spans.
LUNAR_INCLINATION_DEG = 5.145

# Latitudes the horizon quantities are tabulated at. The Veneto sits at 45.5.
DEFAULT_LATITUDES = (0.0, 10.0, 20.0, 30.0, 40.0, 45.5, 50.0, 60.0, 70.0)

# Eccentricity and the longitude of perihelion vary too, but far more slowly than
# the obliquity and with a smaller effect on what is asked here; the present
# values are used for the season lengths and the resulting figure is the shape of
# the variation rather than its exact amplitude at remote epochs.
ECCENTRICITY = 0.016708
PERIHELION_LON_J2000_DEG = 282.937
PERIHELION_PRECESSION_PERIOD_KYR = 20.9
TROPICAL_YEAR_D = 365.2422


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


def obliquity_series(epochs_kyr: np.ndarray) -> np.ndarray:
    """Mean obliquity of date, degrees, from the long-term precession model.

    The angle between the equator and the ecliptic of date is recovered as the
    angle between their pole vectors, which is what ltpequ and ltpecl return.
    """
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


def galactic_pole_of_date(epochs_kyr: np.ndarray):
    """Right ascension and declination of the north galactic pole, of date.

    The galactic plane is fixed in space; what moves is the equatorial frame, so
    the pole is carried from ICRS to the equator of date by the same long-term
    precession matrix used for the stars.
    """
    # ICRS direction of the north galactic pole, IAU 1958 convention.
    ra0, dec0 = math.radians(192.85948), math.radians(27.12825)
    v0 = np.array([math.cos(dec0) * math.cos(ra0),
                   math.cos(dec0) * math.sin(ra0),
                   math.sin(dec0)])
    ra = np.empty(epochs_kyr.size)
    dec = np.empty(epochs_kyr.size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, t in enumerate(epochs_kyr):
            rp = np.asarray(erfa.ltpb(1000.0 * float(t)), dtype=float)
            v = rp @ v0
            ra[i] = math.degrees(math.atan2(v[1], v[0])) % 360.0
            dec[i] = math.degrees(math.asin(np.clip(v[2], -1.0, 1.0)))
    return ra, dec


def rise_azimuth(dec_deg, lat_deg):
    """Azimuth from north at which a body of that declination rises.

    Geometric horizon, no refraction. Returns NaN where the body never rises or
    never sets from that latitude, which is exactly where the notion of a rising
    point stops meaning anything.
    """
    d = np.radians(np.asarray(dec_deg, dtype=float))
    p = math.radians(lat_deg)
    c = np.sin(d) / math.cos(p) if abs(math.cos(p)) > 1e-12 else np.full_like(d, np.nan)
    c = np.where(np.abs(c) <= 1.0, c, np.nan)
    return np.degrees(np.arccos(c))


def season_lengths(epochs_kyr: np.ndarray):
    """Days from each equinox or solstice to the next.

    The Earth runs fastest at perihelion, so the quarter of the orbit containing
    it is crossed in fewest days. Perihelion circles the year in about 21 kyr,
    which is why these four numbers are not constants: today it falls in early
    January and northern winter is the short season, ten thousand years ago it
    fell in July and summer was.

    First order in the eccentricity, which at 0.0167 is ample for the few days of
    variation involved.
    """
    # Longitude of perihelion measured from the vernal equinox, moving backwards
    # through the seasons as the equinox precesses.
    lon_p = np.radians(
        (PERIHELION_LON_J2000_DEG
         - 360.0 * epochs_kyr / PERIHELION_PRECESSION_PERIOD_KYR) % 360.0)

    starts = np.radians([0.0, 90.0, 180.0, 270.0])      # spring, summer, autumn, winter
    out = np.empty((epochs_kyr.size, 4))
    for k, s0 in enumerate(starts):
        s1 = s0 + math.pi / 2.0
        # Time from true longitude s0 to s1, first order in e:
        #   t = (P/2pi) * [ (s1-s0) + 2e (sin(s0-lon_p) - sin(s1-lon_p)) ]
        out[:, k] = (TROPICAL_YEAR_D / (2.0 * math.pi)) * (
            (s1 - s0) + 2.0 * ECCENTRICITY *
            (np.sin(s0 - lon_p) - np.sin(s1 - lon_p)))
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Obliquity and the horizon geometry that follows from it.")
    p.add_argument("--Tmin", type=float, default=-200.0, help="earliest epoch, kyr")
    p.add_argument("--Tmax", type=float, default=2.0, help="latest epoch, kyr")
    p.add_argument("--dt", type=float, default=0.1, help="epoch step, kyr")
    p.add_argument("--latitudes", type=float, nargs="+",
                   default=list(DEFAULT_LATITUDES))
    p.add_argument("--outdir", default="earth")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    n = int(round((args.Tmax - args.Tmin) / abs(args.dt))) + 1
    epochs = args.Tmin + abs(args.dt) * np.arange(n)
    log(f"Epoche: {epochs.size} da {epochs.min():+.1f} a {epochs.max():+.1f} kyr")

    log("Obliquita' dal modello a lungo termine ...")
    eps = obliquity_series(epochs)
    log(f"  intervallo {eps.min():.3f}° .. {eps.max():.3f}°, oggi {eps[-1]:.3f}°")

    log("Polo galattico di data ...")
    gp_ra, gp_dec = galactic_pole_of_date(epochs)

    log("Lunghezze delle stagioni ...")
    seas = season_lengths(epochs)

    core = pd.DataFrame({
        "epoch_kyr": epochs,
        "obliquity_deg": eps,
        "solstice_dec_deg": eps,
        "moon_major_dec_deg": eps + LUNAR_INCLINATION_DEG,
        "moon_minor_dec_deg": eps - LUNAR_INCLINATION_DEG,
        "gal_pole_ra_deg": gp_ra,
        "gal_pole_dec_deg": gp_dec,
        "spring_days": seas[:, 0], "summer_days": seas[:, 1],
        "autumn_days": seas[:, 2], "winter_days": seas[:, 3],
    })
    core.to_csv(outdir / "orientation.csv", index=False, float_format="%.6f")
    log(f"  scritto {outdir}/orientation.csv")

    # Horizon points per latitude. The Sun at the solstices and the Moon at its
    # standstills differ only in the declination fed to the same formula.
    rows = []
    for lat in args.latitudes:
        for name, dec in (("sole_solstizio", eps),
                          ("luna_lunistizio_maggiore", eps + LUNAR_INCLINATION_DEG),
                          ("luna_lunistizio_minore", eps - LUNAR_INCLINATION_DEG)):
            az = rise_azimuth(dec, lat)
            for i in range(epochs.size):
                rows.append({"epoch_kyr": epochs[i], "lat_deg": lat,
                             "body": name, "dec_deg": dec[i],
                             "rise_azimuth_deg": az[i],
                             "set_azimuth_deg": (360.0 - az[i]
                                                 if np.isfinite(az[i]) else np.nan)})
    hz = pd.DataFrame(rows)
    hz.to_csv(outdir / "horizon_points.csv", index=False, float_format="%.4f")
    log(f"  scritto {outdir}/horizon_points.csv ({len(hz):,} righe)")

    log("")
    log("  Escursione dell'azimut di sorgere sull'intero periodo, per latitudine:")
    log(f"    {'lat':>6s} {'solstizio':>22s} {'lunist. magg.':>22s}")
    for lat in args.latitudes:
        s = hz[(hz.lat_deg == lat) & (hz.body == "sole_solstizio")]["rise_azimuth_deg"]
        m = hz[(hz.lat_deg == lat) &
               (hz.body == "luna_lunistizio_maggiore")]["rise_azimuth_deg"]
        def rng(x):
            x = x.dropna()
            return (f"{x.min():5.1f}-{x.max():5.1f}° ({x.max()-x.min():4.1f}°)"
                    if len(x) else "        mai         ")
        log(f"    {lat:+6.1f} {rng(s):>22s} {rng(m):>22s}")

    # ---- figures -------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    axes[0].plot(epochs, eps, lw=1.3, color="darkred")
    axes[0].set_ylabel("obliquita' [°]")
    axes[0].set_title("Inclinazione dell'asse terrestre e cio' che ne discende")
    axes[0].grid(alpha=0.3)

    for lat in args.latitudes:
        s = hz[(hz.lat_deg == lat) & (hz.body == "sole_solstizio")]
        if s["rise_azimuth_deg"].notna().any():
            axes[1].plot(s["epoch_kyr"], s["rise_azimuth_deg"], lw=1.1,
                         label=f"{lat:+.1f}°")
    axes[1].set_ylabel("azimut del sorgere\nal solstizio [° da nord]")
    axes[1].legend(fontsize=7, ncol=5, title="latitudine")
    axes[1].grid(alpha=0.3)

    for k, nome in enumerate(("primavera", "estate", "autunno", "inverno")):
        axes[2].plot(epochs, seas[:, k], lw=1.1, label=nome)
    axes[2].set_ylabel("durata delle stagioni [giorni]")
    axes[2].set_xlabel("Epoca [kyr dall'anno 0]")
    axes[2].legend(fontsize=8, ncol=4)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "orientation.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    # The horizon fan: where Sun and Moon stop, at one latitude, through time.
    lat0 = 45.5 if 45.5 in args.latitudes else args.latitudes[len(args.latitudes) // 2]
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, colour, style in (("luna_lunistizio_maggiore", "navy", "-"),
                                ("sole_solstizio", "darkorange", "-"),
                                ("luna_lunistizio_minore", "teal", "--")):
        s = hz[(hz.lat_deg == lat0) & (hz.body == name)]
        ax.plot(s["epoch_kyr"], s["rise_azimuth_deg"], lw=1.6,
                color=colour, ls=style, label=name.replace("_", " "))
    ax.axhline(90.0, color="k", lw=0.7, ls=":", label="est")
    ax.set_xlabel("Epoca [kyr dall'anno 0]")
    ax.set_ylabel("azimut del sorgere [° da nord]")
    ax.set_title(f"Punti d'arresto sull'orizzonte a {lat0:+.1f}° di latitudine\n"
                 f"il solstizio si sposta di alcuni gradi: un allineamento "
                 f"vale per l'epoca in cui e' stato tracciato")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "horizon_points.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    log("")
    log(f"Scritti {outdir}/orientation.png, horizon_points.png")


if __name__ == "__main__":
    main()
