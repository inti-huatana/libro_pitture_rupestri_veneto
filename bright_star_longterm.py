#!/usr/bin/env python3
"""
bright_star_longterm.py

Long-term linear stellar propagation and visibility phenomena for a bright-star
Hipparcos subset.

Dependencies:
    numpy
    pandas
    scipy
    pyerfa

PyERFA is a Python wrapper of ERFA, which is derived from IAU SOFA.

Time convention
---------------
--Tmin and --Tmax are signed kyr relative to astronomical year 0:
    0      -> astronomical year 0
    -1     -> astronomical year -1000
    +1     -> astronomical year +1000

The output is generated from --Tmax toward --Tmin at steps of |--dt|.

Astrometry
----------
Stars are propagated in a rectilinear heliocentric 3-D model using:
RA, Dec, parallax, proper motion and radial velocity.

Hipparcos pmra is assumed to be mu_alpha*cos(delta), in mas/yr.

Alpha Centauri
--------------
HIP 71681 is removed. HIP 71683 supplies all astrometry and identifiers.
Hp, U, B, V and I magnitudes are replaced by the flux-combined magnitudes of
HIP 71683 + HIP 71681.

Precession and nutation
-----------------------
Long-term precession uses the Vondrak model through ERFA/SOFA LTPB/LTPEQU/
LTPECL. IAU 2000A nutation is optionally superposed (enabled by default).
The long-term Vondrak precession model is intended for +/-200 kyr; IAU 2000A
nutation is not a 200-kyr theory, so its use at very remote epochs is an
extrapolated periodic correction. Use --no-nutation to suppress it.

Heliacal/acronychal model
-------------------------
The event calculation uses a transparent geometric arcus-visionis model:

    AV [deg] = av_a + av_b * V

with defaults av_a=10.5 and av_b=1.4.

For stars that actually rise and set:
  heliacal rising:
      star at rising, Sun altitude = -AV, morning solution
  heliacal setting:
      star at setting, Sun altitude = -AV, evening solution
  acronychal rising:
      star at rising, Sun altitude = 0 deg, evening solution
  acronychal setting:
      star at setting, Sun altitude = 0 deg, morning solution

For INVISIBILE or CIRCUMPOLARE stars these four horizon-crossing events are
undefined and are written as NaN.

Days are measured from the vernal equinox of the epoch:
day 1 = solar ecliptic longitude 0 deg.
A uniform tropical year of 365.2422 d is used, so the output is a season phase,
not a historical calendar date. Refraction and local horizon relief are ignored.
"""

from __future__ import annotations

import argparse
import math
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

try:
    import erfa
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'pyerfa'. Install with: pip install pyerfa"
    ) from exc


MAG_COLS = ("Hpmag", "Umag", "Bmag", "Vmag", "Imag")
KMS_TO_PCYR = 1.0227121650537077e-6
MAS_TO_RAD = np.deg2rad(1.0 / 3_600_000.0)
TROPICAL_YEAR_D = 365.2422

# ICRS -> Galactic rotation matrix, IAU/J2000 convention.
R_ICRS_TO_GAL = np.array(
    [
        [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
        [+0.4941094278755837, -0.4448296299600112, +0.7469822444972189],
        [-0.8676661490190047, -0.1980763734312015, +0.4559837761750669],
    ],
    dtype=float,
)

_CTX = {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Propagate bright Hipparcos stars over +/-200 kyr."
    )
    p.add_argument("--input", required=True, help="Input CSV catalogue.")
    p.add_argument("--outdir", required=True, help="Output directory.")
    p.add_argument("--lat", required=True, type=float,
                   help="Observer geographic latitude in degrees.")
    p.add_argument("--Tmin", required=True, type=float,
                   help="Minimum signed epoch in kyr from astronomical year 0.")
    p.add_argument("--Tmax", required=True, type=float,
                   help="Maximum signed epoch in kyr from astronomical year 0.")
    p.add_argument("--dt", required=True, type=float,
                   help="Time step in kyr; absolute value is used.")
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1),
                   help="Worker processes. Default: all logical CPUs.")
    p.add_argument("--av-a", type=float, default=10.5,
                   help="Arcus visionis intercept in degrees. Default: 10.5.")
    p.add_argument("--av-b", type=float, default=1.4,
                   help="Arcus visionis coefficient in deg/mag. Default: 1.4.")
    p.add_argument("--no-nutation", action="store_true",
                   help="Use Vondrak long-term precession without IAU 2000A nutation.")
    return p.parse_args()


def combine_mag(m1: float, m2: float) -> float:
    if not (np.isfinite(m1) and np.isfinite(m2)):
        return np.nan
    return -2.5 * np.log10(10.0 ** (-0.4 * m1) + 10.0 ** (-0.4 * m2))


def prepare_catalogue(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "HIP", "HD", "Bayer", "NAME", "ra2000", "de2000", "plx",
        "pmra", "pmde", "rv", *MAG_COLS,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    for col in ("HIP", "ra2000", "de2000", "plx", "pmra", "pmde", "rv", *MAG_COLS):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    astrom = ("HIP", "ra2000", "de2000", "plx", "pmra", "pmde", "rv")
    if df[list(astrom)].isna().any().any():
        bad = df.loc[df[list(astrom)].isna().any(axis=1), "HIP"].tolist()
        raise ValueError(f"Null astrometry for HIP entries: {bad[:20]}")

    a = df.loc[df["HIP"] == 71683]
    b = df.loc[df["HIP"] == 71681]
    if len(a) != 1 or len(b) != 1:
        raise ValueError("HIP 71683 and HIP 71681 must both occur exactly once.")

    ia, ib = a.index[0], b.index[0]
    for col in MAG_COLS:
        df.loc[ia, col] = combine_mag(float(df.loc[ia, col]), float(df.loc[ib, col]))

    # Keep HIP 71683 astrometry/identifiers, remove HIP 71681.
    df = df.loc[df["HIP"] != 71681].copy()
    df["HIP"] = df["HIP"].astype(int)
    return df.reset_index(drop=True)


def make_times(tmin: float, tmax: float, dt: float) -> np.ndarray:
    if dt == 0:
        raise ValueError("--dt must be non-zero.")
    if max(abs(tmin), abs(tmax)) > 200.0:
        raise ValueError("|--Tmin| and |--Tmax| must be <= 200 kyr.")

    step = abs(dt)
    direction = -1.0 if tmax >= tmin else +1.0
    stop = tmin + direction * 0.5 * step
    x = np.arange(tmax, stop, direction * step, dtype=float)

    # Ensure the requested endpoint appears if numerically aligned.
    if len(x) == 0 or abs(x[-1] - tmin) > 1e-10:
        if abs((tmax - tmin) / step - round((tmax - tmin) / step)) < 1e-10:
            x = np.append(x, tmin)
    return x


def rot_x_eq_to_ecl(v: np.ndarray, eps: float) -> np.ndarray:
    c, s = math.cos(eps), math.sin(eps)
    out = np.empty_like(v)
    out[..., 0] = v[..., 0]
    out[..., 1] = c * v[..., 1] + s * v[..., 2]
    out[..., 2] = -s * v[..., 1] + c * v[..., 2]
    return out


def precompute_earth_orientation(times_kyr: np.ndarray, use_nutation: bool):
    epj = 1000.0 * times_kyr
    n = len(epj)

    r_true = np.empty((n, 3, 3), dtype=float)
    eps_true = np.empty(n, dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, y in enumerate(epj):
            # LTPB: ICRS/GCRS to mean equator/equinox of date,
            # including frame bias.
            rp = np.asarray(erfa.ltpb(float(y)), dtype=float)

            veq = np.asarray(erfa.ltpequ(float(y)), dtype=float)
            vecl = np.asarray(erfa.ltpecl(float(y)), dtype=float)
            eps_mean = math.acos(np.clip(float(np.dot(veq, vecl)), -1.0, 1.0))

            if use_nutation:
                jd1, jd2 = erfa.epj2jd(float(y))
                dpsi, deps = erfa.nut00a(jd1, jd2)
                rn = np.asarray(erfa.numat(eps_mean, dpsi, deps), dtype=float)
                r_true[i] = rn @ rp
                eps_true[i] = eps_mean + deps
            else:
                r_true[i] = rp
                eps_true[i] = eps_mean

    return epj, r_true, eps_true


def unit_basis(ra: float, dec: float):
    ca, sa = math.cos(ra), math.sin(ra)
    cd, sd = math.cos(dec), math.sin(dec)
    u = np.array([cd * ca, cd * sa, sd])
    ea = np.array([-sa, ca, 0.0])
    ed = np.array([-sd * ca, -sd * sa, cd])
    return u, ea, ed


def vector_to_lonlat(v: np.ndarray):
    lon = np.mod(np.arctan2(v[..., 1], v[..., 0]), 2.0 * np.pi)
    lat = np.arcsin(np.clip(v[..., 2], -1.0, 1.0))
    return np.rad2deg(lon), np.rad2deg(lat)


def wrap_pi(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def visibility_class(dec_rad: float, lat_rad: float) -> str:
    # Geometric horizon, no refraction.
    if abs(lat_rad) >= np.pi / 2:
        same = np.sign(dec_rad) == np.sign(lat_rad)
        return "CIRCUMPOLARE" if same else "INVISIBILE"

    if np.sign(dec_rad) == np.sign(lat_rad) and abs(dec_rad) + abs(lat_rad) > np.pi / 2:
        return "CIRCUMPOLARE"

    if np.sign(dec_rad) != np.sign(lat_rad) and abs(dec_rad) + abs(lat_rad) >= np.pi / 2:
        return "INVISIBILE"

    return "VISIBILE"


def sun_radec_from_lambda(lam: float, eps: float):
    cl, sl = math.cos(lam), math.sin(lam)
    ce, se = math.cos(eps), math.sin(eps)
    x = cl
    y = ce * sl
    z = se * sl
    ra = math.atan2(y, x) % (2.0 * math.pi)
    dec = math.asin(np.clip(z, -1.0, 1.0))
    return ra, dec


def sun_altitude(lam: float, lst: float, lat: float, eps: float):
    ra_s, dec_s = sun_radec_from_lambda(lam, eps)
    h = wrap_pi(lst - ra_s)
    alt = math.asin(
        math.sin(lat) * math.sin(dec_s)
        + math.cos(lat) * math.cos(dec_s) * math.cos(h)
    )
    return alt, h


def roots_over_solar_longitude(lst: float, lat: float, eps: float, target_alt: float):
    # Fine enough to bracket all simple annual roots, followed by Brent refinement.
    grid = np.linspace(0.0, 2.0 * np.pi, 1441)
    vals = np.empty_like(grid)

    for i, lam in enumerate(grid):
        vals[i] = sun_altitude(lam, lst, lat, eps)[0] - target_alt

    roots = []
    for i in range(len(grid) - 1):
        a, b = grid[i], grid[i + 1]
        fa, fb = vals[i], vals[i + 1]

        if fa == 0.0:
            root = a
        elif fa * fb > 0.0:
            continue
        else:
            root = brentq(
                lambda x: sun_altitude(x, lst, lat, eps)[0] - target_alt,
                a, b, xtol=1e-12, rtol=1e-12,
            )

        root %= 2.0 * np.pi
        if not roots or min(abs(wrap_pi(root - r)) for r in roots) > 1e-7:
            roots.append(root)

    return roots


def pick_root_day(roots, lst, lat, eps, morning: bool):
    candidates = []
    for lam in roots:
        _, hsun = sun_altitude(lam, lst, lat, eps)
        if morning and hsun < 0:
            candidates.append(lam)
        elif (not morning) and hsun > 0:
            candidates.append(lam)

    if not candidates:
        return np.nan

    # Normally unique after morning/evening selection.
    lam = candidates[0]
    return 1.0 + (lam / (2.0 * np.pi)) * TROPICAL_YEAR_D


def horizon_events(ra: float, dec: float, lat: float, eps: float, av_deg: float):
    vis = visibility_class(dec, lat)
    if vis != "VISIBILE":
        return vis, np.nan, np.nan, np.nan, np.nan

    cos_h0 = -math.tan(lat) * math.tan(dec)
    if not -1.0 <= cos_h0 <= 1.0:
        return vis, np.nan, np.nan, np.nan, np.nan

    h0 = math.acos(np.clip(cos_h0, -1.0, 1.0))
    lst_rise = (ra - h0) % (2.0 * np.pi)
    lst_set = (ra + h0) % (2.0 * np.pi)

    av = math.radians(av_deg)

    hr_roots = roots_over_solar_longitude(lst_rise, lat, eps, -av)
    hs_roots = roots_over_solar_longitude(lst_set, lat, eps, -av)
    ar_roots = roots_over_solar_longitude(lst_rise, lat, eps, 0.0)
    aset_roots = roots_over_solar_longitude(lst_set, lat, eps, 0.0)

    heliacal_rising = pick_root_day(hr_roots, lst_rise, lat, eps, morning=True)
    heliacal_setting = pick_root_day(hs_roots, lst_set, lat, eps, morning=False)
    acronychal_rising = pick_root_day(ar_roots, lst_rise, lat, eps, morning=False)
    acronychal_setting = pick_root_day(aset_roots, lst_set, lat, eps, morning=True)

    return vis, heliacal_rising, heliacal_setting, acronychal_rising, acronychal_setting


def init_worker(times_kyr, epj, r_true, eps_true, lat_deg, av_a, av_b, outdir):
    global _CTX
    _CTX = {
        "times_kyr": times_kyr,
        "epj": epj,
        "r_true": r_true,
        "eps_true": eps_true,
        "lat": math.radians(lat_deg),
        "av_a": av_a,
        "av_b": av_b,
        "outdir": outdir,
    }


def process_star(record: dict):
    times = _CTX["times_kyr"]
    epj = _CTX["epj"]
    r_true = _CTX["r_true"]
    eps_true = _CTX["eps_true"]
    lat = _CTX["lat"]
    av_a = _CTX["av_a"]
    av_b = _CTX["av_b"]
    outdir = Path(_CTX["outdir"])

    hip = int(record["HIP"])
    ra0 = math.radians(float(record["ra2000"]))
    dec0 = math.radians(float(record["de2000"]))
    plx = float(record["plx"])
    pmra = float(record["pmra"])
    pmde = float(record["pmde"])
    rv = float(record["rv"])

    u0, ea, ed = unit_basis(ra0, dec0)
    d0_pc = 1000.0 / plx

    mu_a = pmra * MAS_TO_RAD
    mu_d = pmde * MAS_TO_RAD

    # Tangential velocity directly from angular motion and distance.
    v_tan_pcyr = d0_pc * (mu_a * ea + mu_d * ed)
    v_rad_pcyr = rv * KMS_TO_PCYR * u0
    v_pcyr = v_tan_pcyr + v_rad_pcyr

    r0_pc = d0_pc * u0

    target_year = epj
    dt_year = target_year - 2000.0
    r_icrs_pc = r0_pc[None, :] + dt_year[:, None] * v_pcyr[None, :]
    dist_pc = np.linalg.norm(r_icrs_pc, axis=1)
    u_icrs = r_icrs_pc / dist_pc[:, None]

    # True equatorial coordinates of date.
    u_true = np.einsum("nij,nj->ni", r_true, u_icrs)
    ra_true, dec_true = vector_to_lonlat(u_true)
    ra_true_rad = np.deg2rad(ra_true)
    dec_true_rad = np.deg2rad(dec_true)

    # True ecliptic coordinates of date.
    u_ecl = np.empty_like(u_true)
    for i in range(len(times)):
        u_ecl[i] = rot_x_eq_to_ecl(u_true[i], float(eps_true[i]))
    ecl_lon, ecl_lat = vector_to_lonlat(u_ecl)

    # Galactic coordinates remain referred to the fixed ICRS Galactic system.
    u_gal = u_icrs @ R_ICRS_TO_GAL.T
    gal_l, gal_b = vector_to_lonlat(u_gal)

    dm = 5.0 * np.log10(dist_pc / d0_pc)
    mags = {}
    for col in MAG_COLS:
        m0 = record[col]
        if pd.isna(m0):
            mags[col] = np.full(len(times), np.nan)
        else:
            mags[col] = float(m0) + dm

    visibility = np.empty(len(times), dtype=object)
    h_rise = np.full(len(times), np.nan)
    h_set = np.full(len(times), np.nan)
    a_rise = np.full(len(times), np.nan)
    a_set = np.full(len(times), np.nan)
    av = _CTX["av_a"] + _CTX["av_b"] * mags["Vmag"]

    for i in range(len(times)):
        (
            visibility[i],
            h_rise[i],
            h_set[i],
            a_rise[i],
            a_set[i],
        ) = horizon_events(
            float(ra_true_rad[i]),
            float(dec_true_rad[i]),
            lat,
            float(eps_true[i]),
            float(av[i]),
        )

    out = pd.DataFrame(
        {
            "epoch_kyr_from_year0": times,
            "astronomical_year": epj,
            "HIP": hip,
            "HD": record.get("HD", np.nan),
            "Bayer": record.get("Bayer", np.nan),
            "NAME": record.get("NAME", np.nan),
            "ra_deg": ra_true,
            "dec_deg": dec_true,
            "ecl_lon_deg": ecl_lon,
            "ecl_lat_deg": ecl_lat,
            "gal_l_deg": gal_l,
            "gal_b_deg": gal_b,
            "distance_pc": dist_pc,
            "Hpmag": mags["Hpmag"],
            "Umag": mags["Umag"],
            "Bmag": mags["Bmag"],
            "Vmag": mags["Vmag"],
            "Imag": mags["Imag"],
            "visibility": visibility,
            "heliacal_rising_day": h_rise,
            "heliacal_setting_day": h_set,
            "acronychal_rising_day": a_rise,
            "acronychal_setting_day": a_set,
            "arcus_visionis_deg": av,
        }
    )

    outpath = outdir / f"HIP_{hip}.csv"
    out.to_csv(outpath, index=False, float_format="%.8f")
    return hip, str(outpath)


def main():
    args = parse_args()

    if not -90.0 <= args.lat <= 90.0:
        raise SystemExit("--lat must be within [-90, +90] deg.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = prepare_catalogue(args.input)
    times = make_times(args.Tmin, args.Tmax, args.dt)

    epj, r_true, eps_true = precompute_earth_orientation(
        times, use_nutation=not args.no_nutation
    )

    records = df.to_dict("records")
    workers = max(1, int(args.workers))

    print(f"Stars: {len(records)}")
    print(f"Epochs per star: {len(times)}")
    print(f"Workers: {workers}")
    print(f"Nutation: {'off' if args.no_nutation else 'IAU 2000A extrapolated'}")
    print(f"Output: {outdir}")

    done = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(
            times, epj, r_true, eps_true, args.lat,
            args.av_a, args.av_b, str(outdir),
        ),
    ) as pool:
        futures = [pool.submit(process_star, rec) for rec in records]
        for fut in as_completed(futures):
            hip, path = fut.result()
            done += 1
            if done % 25 == 0 or done == len(records):
                print(f"{done}/{len(records)} completed")

    print("Done.")


if __name__ == "__main__":
    main()
