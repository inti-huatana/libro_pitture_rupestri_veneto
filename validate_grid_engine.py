#!/usr/bin/env python3
"""
validate_grid_engine.py

Check that the vectorized horizon-event engine of bright_star_grid.py returns
the same visibility classes and event days as the scalar engine of
bright_star_longterm.py, which it replaces.

The two implementations differ only in how the roots of
(solar altitude - target) are located and refined: the old one scans LAM_GRID
in a Python loop and refines with scipy brentq, the new one brackets the same
grid cells with vectorized sign tests and refines by fixed-count bisection.
Everything upstream (catalogue preparation, propagation, precession, solar
grid) is shared code, so the comparison isolates the event search.

Usage:
    python3 validate_grid_engine.py --input hip_mag4.csv [--stars 40] [--epochs 12]
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="hip_mag4.csv")
    p.add_argument("--stars", type=int, default=40, help="Random stars to test.")
    p.add_argument("--epochs", type=int, default=12, help="Epochs to test.")
    p.add_argument("--seed", type=int, default=20260814)
    p.add_argument("--tol-day", type=float, default=1e-6,
                   help="Max tolerated difference in event day.")
    args = p.parse_args()

    sys.argv = [sys.argv[0]]
    old = load_module("bright_star_longterm.py", "bsl_old")
    new = load_module("bright_star_grid.py", "bsl_new")

    rng = np.random.default_rng(args.seed)

    cat = new.prepare_catalogue(args.input)
    idx = rng.choice(len(cat), size=min(args.stars, len(cat)), replace=False)
    sample = cat.iloc[np.sort(idx)]

    # A time span wide enough to exercise precession, at a coarse step.
    times = new.make_times(-50.0, 0.0, 50.0 / (args.epochs - 1))
    epj, r_true, eps_true = new.precompute_earth_orientation(times, use_nutation=False)
    sun_ra, sin_ds, cos_ds = new.precompute_sun_grid(eps_true)

    # The old engine wants Dec on the grid, the new one sin/cos of it.
    sun_dec = np.arcsin(np.clip(sin_ds, -1.0, 1.0))

    lats = new.make_latitudes(70.0, -60.0, 10.0)

    print(f"stars {len(sample)} | epochs {times.size} | latitudes {lats.size} "
          f"| comparisons {len(sample) * times.size * lats.size:,}")

    fields = ["heliacal_rising", "heliacal_setting",
              "acronychal_rising", "acronychal_setting"]
    worst = {f: 0.0 for f in fields}
    worst_where = {f: None for f in fields}
    n_vis_mismatch = 0
    n_nan_mismatch = 0
    n_compared = 0
    n_both_nan = 0

    for rec in sample.to_dict("records"):
        traj = new.compute_trajectory(rec, epj, r_true, eps_true)
        ra_rad = np.deg2rad(traj["ra_deg"])
        dec_rad = np.deg2rad(traj["dec_deg"])
        av_deg = 10.5 + 1.4 * traj["Vmag"]

        for lat_deg in lats:
            vis_n, hr_n, hs_n, ar_n, as_n = new.events_for_latitude(
                ra_rad, dec_rad, av_deg, float(lat_deg),
                eps_true, sun_ra, sin_ds, cos_ds)
            new_vals = {"heliacal_rising": hr_n, "heliacal_setting": hs_n,
                        "acronychal_rising": ar_n, "acronychal_setting": as_n}

            lat_rad = math.radians(float(lat_deg))
            for i in range(times.size):
                res = old.horizon_events(
                    float(ra_rad[i]), float(dec_rad[i]), lat_rad,
                    float(eps_true[i]), float(av_deg[i]),
                    sun_ra[i], sun_dec[i])
                vis_o = res[0]
                old_vals = dict(zip(fields, res[1:]))

                if vis_o != vis_n[i]:
                    n_vis_mismatch += 1
                    print(f"  VIS  HIP {rec['HIP']} lat {lat_deg:+.0f} "
                          f"epoch {times[i]:+.1f}: old={vis_o} new={vis_n[i]}")

                for f in fields:
                    o, nv = old_vals[f], new_vals[f][i]
                    if np.isnan(o) and np.isnan(nv):
                        n_both_nan += 1
                        continue
                    if np.isnan(o) != np.isnan(nv):
                        n_nan_mismatch += 1
                        if n_nan_mismatch <= 10:
                            print(f"  NAN  HIP {rec['HIP']} lat {lat_deg:+.0f} "
                                  f"epoch {times[i]:+.1f} {f}: old={o} new={nv}")
                        continue
                    n_compared += 1
                    d = abs(o - nv)
                    if d > worst[f]:
                        worst[f] = d
                        worst_where[f] = (int(rec["HIP"]), float(lat_deg),
                                          float(times[i]), o, nv)

    print()
    print(f"visibility mismatches : {n_vis_mismatch}")
    print(f"NaN-pattern mismatches: {n_nan_mismatch}")
    print(f"numeric comparisons   : {n_compared:,}   (both NaN: {n_both_nan:,})")
    print()
    ok = (n_vis_mismatch == 0 and n_nan_mismatch == 0)
    for f in fields:
        w = worst[f]
        flag = "OK" if w <= args.tol_day else "FAIL"
        if w > args.tol_day:
            ok = False
        print(f"  max |delta| {f:20s} = {w:.3e} d   [{flag}]")
        if worst_where[f] and w > 0:
            hip, la, t, o, nv = worst_where[f]
            print(f"      at HIP {hip}, lat {la:+.0f}, epoch {t:+.1f}: "
                  f"old={o:.10f} new={nv:.10f}")

    print()
    print("RESULT:", "engines agree" if ok else "ENGINES DIFFER")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
