#!/usr/bin/env python3
"""
bright_star_grid.py

Long-term stellar propagation and visibility phenomena for a bright-star
Hipparcos subset, computed once over a grid of observer latitudes.

Rewrite of bright_star_longterm.py with the same physics but a different
work decomposition:

  * stellar trajectories (position, distance, magnitudes) do not depend on the
    observer, so they are computed once per star and reused for every latitude;
  * the horizon-event search, which does depend on latitude, is vectorized over
    all epochs at once instead of running a scalar root-finder in a Python loop.

Physics, conventions and the arcus-visionis model are unchanged with respect to
bright_star_longterm.py; see that file's docstring. Equivalence of the two
implementations is checked by validate_grid_engine.py.

Time convention
---------------
--Tmin and --Tmax are signed kyr relative to astronomical year 0:
    0  -> astronomical year 0
    -1 -> astronomical year -1000
    +1 -> astronomical year +1000

Latitude grid
-------------
--lat-max, --lat-min and --lat-step define the observer latitudes in degrees,
from north to south. Default: +70 to -60 in steps of 5 degrees (27 latitudes).

Output
------
    <outdir>/trajectories.csv    one row per (star, epoch); latitude-independent
    <outdir>/stars/HIP_<n>.csv   one file per star, all latitudes, long format
    <outdir>/fullcat.csv         concatenation of the per-star files (--no-fullcat
                                 to skip; it is the largest output by far)

Dependencies: numpy, pandas, pyerfa.
PyERFA is a Python wrapper of ERFA, which is derived from IAU SOFA.
"""

from __future__ import annotations

import argparse
import math
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd

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
TWO_PI = 2.0 * np.pi

# Solar-longitude scan grid used to bracket the horizon-event roots. Identical
# to the grid in bright_star_longterm.py, so the bracketing is the same.
LAM_GRID = np.linspace(0.0, TWO_PI, 1441)

# Bisection steps used to refine a bracketed root. The initial bracket is one
# grid cell, 2*pi/1440 = 4.4e-3 rad, i.e. 0.25 d once converted to a day of the
# year; 40 halvings take that to 2e-13 d, at the double-precision floor of a
# quantity of order 365 and far below the accuracy of the visibility model
# itself. Measured agreement with the brentq refinement it replaces is ~1e-10 d
# (validate_grid_engine.py).
N_BISECT = 40

# ICRS -> Galactic rotation matrix, IAU/J2000 convention.
R_ICRS_TO_GAL = np.array(
    [
        [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
        [+0.4941094278755837, -0.4448296299600112, +0.7469822444972189],
        [-0.8676661490190047, -0.1980763734312015, +0.4559837761750669],
    ],
    dtype=float,
)

OUT_COLS = [
    "epoch_kyr_from_year0", "astronomical_year", "HIP", "HD", "Bayer", "NAME",
    "lat_deg", "ra_deg", "dec_deg", "ecl_lon_deg", "ecl_lat_deg",
    "gal_l_deg", "gal_b_deg", "distance_pc",
    "Hpmag", "Umag", "Bmag", "Vmag", "Imag",
    "visibility", "heliacal_rising_day", "heliacal_setting_day",
    "acronychal_rising_day", "acronychal_setting_day", "arcus_visionis_deg",
]

_CTX: dict = {}


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Propagate bright Hipparcos stars over a latitude grid."
    )
    p.add_argument("--input", required=True, help="Input Hipparcos subset CSV.")
    p.add_argument("--outdir", required=True, help="Output directory.")
    p.add_argument("--Tmin", type=float, default=-50.0, help="Min epoch, kyr from year 0.")
    p.add_argument("--Tmax", type=float, default=0.0, help="Max epoch, kyr from year 0.")
    p.add_argument("--dt", type=float, default=0.1, help="Epoch step, kyr.")
    p.add_argument("--lat-max", type=float, default=70.0, help="Northernmost latitude, deg.")
    p.add_argument("--lat-min", type=float, default=-60.0, help="Southernmost latitude, deg.")
    p.add_argument("--lat-step", type=float, default=5.0, help="Latitude step, deg.")
    p.add_argument("--workers", type=int, default=os.cpu_count(),
                   help="Process pool size.")
    p.add_argument("--av-a", type=float, default=10.5, help="Arcus visionis intercept, deg.")
    p.add_argument("--av-b", type=float, default=1.4, help="Arcus visionis slope, deg/mag.")
    p.add_argument("--no-nutation", action="store_true",
                   help="Disable the IAU 2000A nutation correction.")
    p.add_argument("--no-fullcat", action="store_true",
                   help="Skip writing the concatenated fullcat.csv.")
    p.add_argument("--float-format", default="%.8f", help="CSV float format.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------
def combine_mag(m1: float, m2: float) -> float:
    if pd.isna(m1) and pd.isna(m2):
        return np.nan
    if pd.isna(m1):
        return float(m2)
    if pd.isna(m2):
        return float(m1)
    return -2.5 * math.log10(10.0 ** (-0.4 * m1) + 10.0 ** (-0.4 * m2))


def prepare_catalogue(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = ["HIP", "ra2000", "de2000", "plx", "pmra", "pmde", "rv"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Input is missing required columns: {missing}")

    for col in MAG_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[np.isfinite(df["plx"]) & (df["plx"] > 0.0)].copy()
    df["rv"] = df["rv"].fillna(0.0)

    # Alpha Centauri: HIP 71683 (A) carries the astrometry and identifiers;
    # the magnitudes become the combined flux of A + B, and B is dropped.
    if (df["HIP"] == 71683).any() and (df["HIP"] == 71681).any():
        a = df.index[df["HIP"] == 71683][0]
        b = df.index[df["HIP"] == 71681][0]
        for col in MAG_COLS:
            df.at[a, col] = combine_mag(df.at[a, col], df.at[b, col])
        df = df.drop(index=b)

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

    if len(x) == 0 or abs(x[-1] - tmin) > 1e-10:
        if abs((tmax - tmin) / step - round((tmax - tmin) / step)) < 1e-10:
            x = np.append(x, tmin)
    return x


def make_latitudes(lat_max: float, lat_min: float, lat_step: float) -> np.ndarray:
    if lat_step <= 0:
        raise ValueError("--lat-step must be positive.")
    if lat_max < lat_min:
        raise ValueError("--lat-max must be >= --lat-min.")
    n = int(round((lat_max - lat_min) / lat_step))
    lats = lat_max - lat_step * np.arange(n + 1, dtype=float)
    if abs(lats[-1] - lat_min) > 1e-9:
        lats = np.append(lats, lat_min)
    return lats


# ---------------------------------------------------------------------------
# Earth orientation and solar grid (shared by every star and every latitude)
# ---------------------------------------------------------------------------
def precompute_earth_orientation(times_kyr: np.ndarray, use_nutation: bool):
    epj = 1000.0 * times_kyr
    n = len(epj)

    r_true = np.empty((n, 3, 3), dtype=float)
    eps_true = np.empty(n, dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, y in enumerate(epj):
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


def precompute_sun_grid(eps_true: np.ndarray):
    """Solar RA and Dec over LAM_GRID for every epoch.

    Depends only on the obliquity, so it is latitude- and star-independent and
    is computed once for the whole run. sin/cos of the declination are returned
    already evaluated because the event search needs them, never Dec itself.
    """
    lam = LAM_GRID[None, :]
    eps = eps_true[:, None]
    cl, sl = np.cos(lam), np.sin(lam)
    ce, se = np.cos(eps), np.sin(eps)
    sun_ra = np.mod(np.arctan2(ce * sl, np.broadcast_to(cl, (eps.shape[0], LAM_GRID.size))), TWO_PI)
    z = np.clip(se * sl, -1.0, 1.0)
    sin_dec = z
    cos_dec = np.sqrt(1.0 - z * z)
    return sun_ra, sin_dec, cos_dec


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def unit_basis(ra: float, dec: float):
    ca, sa = math.cos(ra), math.sin(ra)
    cd, sd = math.cos(dec), math.sin(dec)
    u = np.array([cd * ca, cd * sa, sd])
    ea = np.array([-sa, ca, 0.0])
    ed = np.array([-sd * ca, -sd * sa, cd])
    return u, ea, ed


def vector_to_lonlat(v: np.ndarray):
    lon = np.mod(np.arctan2(v[..., 1], v[..., 0]), TWO_PI)
    lat = np.arcsin(np.clip(v[..., 2], -1.0, 1.0))
    return np.rad2deg(lon), np.rad2deg(lat)


def wrap_pi(x):
    return (x + np.pi) % TWO_PI - np.pi


def rot_x_eq_to_ecl_vec(v: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """Equatorial -> ecliptic rotation about the x axis, vectorized over epochs."""
    c, s = np.cos(eps), np.sin(eps)
    out = np.empty_like(v)
    out[:, 0] = v[:, 0]
    out[:, 1] = c * v[:, 1] + s * v[:, 2]
    out[:, 2] = -s * v[:, 1] + c * v[:, 2]
    return out


def visibility_class_vec(dec_rad: np.ndarray, lat_rad: float) -> np.ndarray:
    """Geometric horizon classification, no refraction. Vectorized over epochs.

    Reproduces visibility_class() of bright_star_longterm.py term by term,
    including its use of np.sign, which treats a declination or latitude of
    exactly zero as matching neither hemisphere.
    """
    out = np.full(dec_rad.shape, "VISIBILE", dtype=object)

    if abs(lat_rad) >= np.pi / 2:
        same = np.sign(dec_rad) == np.sign(lat_rad)
        out[same] = "CIRCUMPOLARE"
        out[~same] = "INVISIBILE"
        return out

    same = np.sign(dec_rad) == np.sign(lat_rad)
    total = np.abs(dec_rad) + abs(lat_rad)
    out[same & (total > np.pi / 2)] = "CIRCUMPOLARE"
    out[(~same) & (total >= np.pi / 2)] = "INVISIBILE"
    return out


# ---------------------------------------------------------------------------
# Horizon events, vectorized over epochs
# ---------------------------------------------------------------------------
def _sun_sinalt_hour_angle(lam, eps, lst, sin_lat, cos_lat):
    """sin(solar altitude) and hour angle at ecliptic longitude lam. Elementwise.

    Returns the sine of the altitude rather than the altitude: the event search
    only ever compares the altitude against a target, and arcsin is strictly
    monotonic, so comparing sines locates exactly the same roots without paying
    for the inverse trigonometric call. See solve_event_day.
    """
    cl, sl = np.cos(lam), np.sin(lam)
    ce, se = np.cos(eps), np.sin(eps)
    ra_s = np.arctan2(ce * sl, cl)
    z = np.clip(se * sl, -1.0, 1.0)
    cos_ds = np.sqrt(1.0 - z * z)
    # cos() is even and 2*pi-periodic, so the hour angle needs no wrapping here;
    # it is wrapped below only where its sign is required.
    sin_alt = sin_lat * z + cos_lat * cos_ds * np.cos(lst - ra_s)
    return sin_alt, wrap_pi(lst - ra_s)


def sin_altitude_on_grid(lst_a, sun_ra_a, sin_ds_a, cos_ds_a, sin_lat, cos_lat):
    """sin(solar altitude) over LAM_GRID, per active epoch. Shape (n_active, G).

    This is the dominant cost of the whole program: it touches n_active * 1441
    elements per call. It depends on the star only through lst_a, and on the
    event only through which sidereal time is used, so it is evaluated twice per
    star per latitude (rising and setting) and shared between the heliacal and
    acronychal searches, which differ only in the target altitude.

    Neither arcsin nor a modulo appears here, for the reasons given in
    _sun_sinalt_hour_angle; both were measured to dominate the runtime.
    """
    return sin_lat * sin_ds_a + cos_lat * cos_ds_a * np.cos(lst_a[:, None] - sun_ra_a)


def solve_event_day(sin_alt_grid, lst_a, sin_tgt_a, eps_a, active_idx, n_epoch,
                    sin_lat, cos_lat, morning: bool) -> np.ndarray:
    """First solar longitude where the Sun reaches the target altitude.

    Brackets every sign change of (sin altitude - sin target) on LAM_GRID,
    refines all brackets at once by bisection, keeps the roots whose hour angle
    has the morning or evening sign, and returns the earliest one in grid order
    -- the same selection rule as pick_root_day() in bright_star_longterm.py.

    Working in the sine of the altitude rather than the altitude itself is exact,
    not an approximation: arcsin is strictly increasing on [-1, 1], so it
    preserves both the location and the sign pattern of every root. Values
    outside [-1, 1], which arcsin would have clipped, keep the correct sign too.
    """
    days = np.full(n_epoch, np.nan)
    if active_idx.size == 0:
        return days

    vals = sin_alt_grid - sin_tgt_a[:, None]
    fa = vals[:, :-1]
    fb = vals[:, 1:]
    cross = (fa * fb <= 0.0) & ~((fa == 0.0) & (fb == 0.0))

    ri, gi = np.where(cross)          # row-major: sorted by row, then by grid cell
    if ri.size == 0:
        return days

    a = LAM_GRID[gi].copy()
    b = LAM_GRID[gi + 1].copy()
    fa_v = vals[ri, gi].copy()
    eps_r = eps_a[ri]
    lst_r = lst_a[ri]
    tgt_r = sin_tgt_a[ri]

    for _ in range(N_BISECT):
        mid = 0.5 * (a + b)
        fm = _sun_sinalt_hour_angle(mid, eps_r, lst_r, sin_lat, cos_lat)[0] - tgt_r
        same = np.sign(fm) == np.sign(fa_v)
        a = np.where(same, mid, a)
        fa_v = np.where(same, fm, fa_v)
        b = np.where(same, b, mid)

    root = 0.5 * (a + b)
    _, h_root = _sun_sinalt_hour_angle(root, eps_r, lst_r, sin_lat, cos_lat)
    ok = h_root < 0.0 if morning else h_root > 0.0
    if not ok.any():
        return days

    ri_ok = ri[ok]
    root_ok = np.mod(root[ok], TWO_PI)
    uniq, first = np.unique(ri_ok, return_index=True)
    days[active_idx[uniq]] = 1.0 + (root_ok[first] / TWO_PI) * TROPICAL_YEAR_D
    return days


def events_for_latitude(ra_rad, dec_rad, av_deg, lat_deg,
                        eps_true, sun_ra, sin_ds, cos_ds):
    """All four horizon events for one star at one latitude, all epochs."""
    n_epoch = ra_rad.size
    lat = math.radians(lat_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)

    visibility = visibility_class_vec(dec_rad, lat)

    cos_h0 = -math.tan(lat) * np.tan(dec_rad)
    active = (visibility == "VISIBILE") & (cos_h0 >= -1.0) & (cos_h0 <= 1.0)
    active_idx = np.where(active)[0]

    nan = np.full(n_epoch, np.nan)
    if active_idx.size == 0:
        return visibility, nan, nan.copy(), nan.copy(), nan.copy()

    h0 = np.arccos(np.clip(cos_h0[active_idx], -1.0, 1.0))
    lst_rise = np.mod(ra_rad[active_idx] - h0, TWO_PI)
    lst_set = np.mod(ra_rad[active_idx] + h0, TWO_PI)

    eps_a = eps_true[active_idx]
    sun_ra_a = sun_ra[active_idx]
    sin_ds_a = sin_ds[active_idx]
    cos_ds_a = cos_ds[active_idx]

    # Targets enter only through their sine; see solve_event_day.
    sin_av = np.sin(-np.radians(av_deg[active_idx]))
    zero = np.zeros_like(sin_av)

    # Two tables, reused by the heliacal and acronychal searches.
    sa_rise = sin_altitude_on_grid(lst_rise, sun_ra_a, sin_ds_a, cos_ds_a, sin_lat, cos_lat)
    sa_set = sin_altitude_on_grid(lst_set, sun_ra_a, sin_ds_a, cos_ds_a, sin_lat, cos_lat)

    h_rise = solve_event_day(sa_rise, lst_rise, sin_av, eps_a, active_idx,
                             n_epoch, sin_lat, cos_lat, morning=True)
    h_set = solve_event_day(sa_set, lst_set, sin_av, eps_a, active_idx,
                            n_epoch, sin_lat, cos_lat, morning=False)
    a_rise = solve_event_day(sa_rise, lst_rise, zero, eps_a, active_idx,
                             n_epoch, sin_lat, cos_lat, morning=False)
    a_set = solve_event_day(sa_set, lst_set, zero, eps_a, active_idx,
                            n_epoch, sin_lat, cos_lat, morning=True)

    return visibility, h_rise, h_set, a_rise, a_set


# ---------------------------------------------------------------------------
# Trajectory (latitude-independent)
# ---------------------------------------------------------------------------
def compute_trajectory(record: dict, epj, r_true, eps_true):
    """Rectilinear 3-D propagation and coordinates of date for one star."""
    ra0 = math.radians(float(record["ra2000"]))
    dec0 = math.radians(float(record["de2000"]))
    plx = float(record["plx"])

    u0, ea, ed = unit_basis(ra0, dec0)
    d0_pc = 1000.0 / plx

    v_tan = d0_pc * (float(record["pmra"]) * MAS_TO_RAD * ea
                     + float(record["pmde"]) * MAS_TO_RAD * ed)
    v_pcyr = v_tan + float(record["rv"]) * KMS_TO_PCYR * u0

    r_icrs = (d0_pc * u0)[None, :] + (epj - 2000.0)[:, None] * v_pcyr[None, :]
    dist_pc = np.linalg.norm(r_icrs, axis=1)
    u_icrs = r_icrs / dist_pc[:, None]

    u_true = np.einsum("nij,nj->ni", r_true, u_icrs)
    ra_deg, dec_deg = vector_to_lonlat(u_true)

    u_ecl = rot_x_eq_to_ecl_vec(u_true, eps_true)
    ecl_lon, ecl_lat = vector_to_lonlat(u_ecl)

    u_gal = u_icrs @ R_ICRS_TO_GAL.T
    gal_l, gal_b = vector_to_lonlat(u_gal)

    dm = 5.0 * np.log10(dist_pc / d0_pc)
    mags = {}
    for col in MAG_COLS:
        m0 = record[col]
        mags[col] = (np.full(dist_pc.size, np.nan) if pd.isna(m0)
                     else float(m0) + dm)

    return {
        "ra_deg": ra_deg, "dec_deg": dec_deg,
        "ecl_lon_deg": ecl_lon, "ecl_lat_deg": ecl_lat,
        "gal_l_deg": gal_l, "gal_b_deg": gal_b,
        "distance_pc": dist_pc, **mags,
    }


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def init_worker(times_kyr, epj, r_true, eps_true, sun_ra, sin_ds, cos_ds,
                lats, av_a, av_b, stars_dir, float_format):
    _CTX.update(
        times_kyr=times_kyr, epj=epj, r_true=r_true, eps_true=eps_true,
        sun_ra=sun_ra, sin_ds=sin_ds, cos_ds=cos_ds, lats=lats,
        av_a=av_a, av_b=av_b, stars_dir=stars_dir, float_format=float_format,
    )


def process_star(record: dict):
    times = _CTX["times_kyr"]
    epj = _CTX["epj"]
    eps_true = _CTX["eps_true"]
    lats = _CTX["lats"]
    n_epoch = times.size

    traj = compute_trajectory(record, epj, _CTX["r_true"], eps_true)
    ra_rad = np.deg2rad(traj["ra_deg"])
    dec_rad = np.deg2rad(traj["dec_deg"])
    av_deg = _CTX["av_a"] + _CTX["av_b"] * traj["Vmag"]

    hip = int(record["HIP"])
    blocks = []
    for lat_deg in lats:
        vis, hr, hs, ar, aset = events_for_latitude(
            ra_rad, dec_rad, av_deg, float(lat_deg), eps_true,
            _CTX["sun_ra"], _CTX["sin_ds"], _CTX["cos_ds"],
        )
        blocks.append(pd.DataFrame({
            "epoch_kyr_from_year0": times,
            "astronomical_year": epj,
            "HIP": hip,
            "HD": record.get("HD", np.nan),
            "Bayer": record.get("Bayer", np.nan),
            "NAME": record.get("NAME", np.nan),
            "lat_deg": float(lat_deg),
            "ra_deg": traj["ra_deg"], "dec_deg": traj["dec_deg"],
            "ecl_lon_deg": traj["ecl_lon_deg"], "ecl_lat_deg": traj["ecl_lat_deg"],
            "gal_l_deg": traj["gal_l_deg"], "gal_b_deg": traj["gal_b_deg"],
            "distance_pc": traj["distance_pc"],
            "Hpmag": traj["Hpmag"], "Umag": traj["Umag"], "Bmag": traj["Bmag"],
            "Vmag": traj["Vmag"], "Imag": traj["Imag"],
            "visibility": vis,
            "heliacal_rising_day": hr, "heliacal_setting_day": hs,
            "acronychal_rising_day": ar, "acronychal_setting_day": aset,
            "arcus_visionis_deg": av_deg,
        }))

    out = pd.concat(blocks, ignore_index=True)[OUT_COLS]
    path = Path(_CTX["stars_dir"]) / f"HIP_{hip}.csv"
    out.to_csv(path, index=False, float_format=_CTX["float_format"])
    return hip, n_epoch * len(lats)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def write_trajectories(cat, times, epj, r_true, eps_true, path, float_format):
    """One row per (star, epoch): everything that does not depend on latitude."""
    frames = []
    for rec in cat.to_dict("records"):
        traj = compute_trajectory(rec, epj, r_true, eps_true)
        frames.append(pd.DataFrame({
            "epoch_kyr_from_year0": times,
            "astronomical_year": epj,
            "HIP": int(rec["HIP"]),
            "HD": rec.get("HD", np.nan),
            "Bayer": rec.get("Bayer", np.nan),
            "NAME": rec.get("NAME", np.nan),
            **traj,
        }))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(path, index=False, float_format=float_format)
    return len(df)


def concat_fullcat(stars_dir: Path, out_path: Path) -> int:
    """Stream the per-star files into one catalogue without loading them all."""
    files = sorted(stars_dir.glob("HIP_*.csv"))
    n = 0
    with out_path.open("w", encoding="utf-8") as dst:
        for i, f in enumerate(files):
            with f.open("r", encoding="utf-8") as src:
                header = src.readline()
                if i == 0:
                    dst.write(header)
                for line in src:
                    dst.write(line)
                    n += 1
    return n


def main() -> None:
    args = parse_args()

    outdir = Path(args.outdir)
    stars_dir = outdir / "stars"
    stars_dir.mkdir(parents=True, exist_ok=True)

    cat = prepare_catalogue(args.input)
    times = make_times(args.Tmin, args.Tmax, args.dt)
    lats = make_latitudes(args.lat_max, args.lat_min, args.lat_step)

    log(f"Stars: {len(cat)} | epochs: {times.size} "
        f"[{times[-1]:+.1f} .. {times[0]:+.1f}] kyr, dt={abs(args.dt)}")
    log(f"Latitudes: {lats.size} [{lats[0]:+.0f} .. {lats[-1]:+.0f}] "
        f"step {args.lat_step}")
    log(f"Output rows expected: {len(cat) * times.size * lats.size:,}")

    log("Precomputing Earth orientation ...")
    t0 = time.perf_counter()
    epj, r_true, eps_true = precompute_earth_orientation(
        times, use_nutation=not args.no_nutation)
    log(f"  done in {time.perf_counter() - t0:.1f} s")

    log("Precomputing solar grid ...")
    t0 = time.perf_counter()
    sun_ra, sin_ds, cos_ds = precompute_sun_grid(eps_true)
    log(f"  done in {time.perf_counter() - t0:.1f} s "
        f"({sun_ra.nbytes * 3 / 2**20:.0f} MB shared)")

    log("Writing trajectories.csv ...")
    t0 = time.perf_counter()
    n_traj = write_trajectories(cat, times, epj, r_true, eps_true,
                                outdir / "trajectories.csv", args.float_format)
    log(f"  {n_traj:,} rows in {time.perf_counter() - t0:.1f} s")

    records = cat.to_dict("records")
    log(f"Computing horizon events with {args.workers} workers ...")
    t0 = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(times, epj, r_true, eps_true, sun_ra, sin_ds, cos_ds,
                  lats, args.av_a, args.av_b, str(stars_dir), args.float_format),
    ) as pool:
        futures = [pool.submit(process_star, r) for r in records]
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if done % 25 == 0 or done == len(records):
                el = time.perf_counter() - t0
                eta = el / done * (len(records) - done)
                log(f"  {done}/{len(records)} stars | elapsed {el:6.1f} s | ETA {eta:6.1f} s")

    log(f"Events done in {time.perf_counter() - t0:.1f} s")

    if not args.no_fullcat:
        log("Concatenating fullcat.csv ...")
        t0 = time.perf_counter()
        n = concat_fullcat(stars_dir, outdir / "fullcat.csv")
        size = (outdir / "fullcat.csv").stat().st_size / 2**20
        log(f"  {n:,} rows, {size:.0f} MB, in {time.perf_counter() - t0:.1f} s")

    log(f"Output in {outdir}/")


if __name__ == "__main__":
    main()
