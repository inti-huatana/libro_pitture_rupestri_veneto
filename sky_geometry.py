#!/usr/bin/env python3
"""
sky_geometry.py

The five reference marks of the sky, and where they went.

Three programs in this set study the celestial pole. None studies the other
circle, and the other circle is arguably the more useful of the two.

    the east star   A star with declination zero rises exactly east and sets
                    exactly west, and it does so from every latitude. That is
                    the property: a pole star must be re-read at each degree of
                    travel, because its altitude is the latitude, while a star
                    on the equator gives the same bearing to everyone. For a
                    population on the move it is worth more than a pole star,
                    and no one has asked which star held the office. The check
                    is immediate and rather beautiful: today it is Mintaka, at
                    declination -0.30 degrees, the western star of Orion's Belt
                    -- the one asterism recognised everywhere on Earth.

    galactic centre Its ecliptic latitude is fixed at -5.6 degrees, so
                    precession swings its declination between -29.0 and +17.8.
                    It is at -29.0 now: we live at the bottom of the cycle. The
                    brightest and most structured part of the Milky Way, the
                    Sagittarius star clouds, culminates 15 degrees above the
                    horizon from the Veneto and is lost in haze; half a
                    precession cycle away it would stand at 62. Over 200 kyr
                    that happens eight times. Nothing else in these data changes
                    the look of the sky by so much.

    ecliptic pole   The point that does not move. The celestial pole runs a
                    circle of 47 degrees across in 26 kyr; the pole of the
                    ecliptic sits still, in Draco, and has no bright star near
                    it. The one fixed point of the firmament is empty, and the
                    conspicuous one is the moving one.

    vernal point    Where the equator cuts the ecliptic. It goes round the sky
                    in 26 kyr, so any scheme built on which stars stand near it
                    resets nearly eight times over the span computed here, and
                    carries no memory beyond one turn.

    the balance     The bright sky is not shared equally. Canopus, Alpha
                    Centauri, Achernar, Crux and Hadar are all southern, so a
                    migration southward gains sky and one northward loses it.
                    That asymmetry is not a constant either.

The last of these questions is the one asked directly: when, and from where,
did the galactic centre pass overhead in a dark sky? Passing overhead alone is
only the condition latitude = declination, a curve in time and latitude. But
the centre culminates at midnight on one date a year, and precession walks that
date around the calendar with the same 26 kyr period, so at some epochs it
comes to the meridian in the short nights of summer and at others in the long
ones of winter. The two conditions are therefore combined here into the hours
per year during which the centre stands high and the sky is genuinely dark,
which is the quantity that answers the question as it was meant.

Usage
-----
    python3 sky_geometry.py --trajectories BSGRID/trajectories.csv \
            --lat 45.5 --outdir geometry
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
from night_visibility import (half_arc, arc_overlap, year_weights,
                              sun_position, obliquity_series)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import erfa
except ImportError as exc:                                    # pragma: no cover
    raise SystemExit("Missing dependency 'pyerfa'. Install with: "
                     "pip install pyerfa") from exc

# Sagittarius A*, ICRS. The dynamical centre of the Galaxy, and within a degree
# of the visual centre of the Sagittarius star clouds.
GC_RA_ICRS_DEG = 266.41684
GC_DEC_ICRS_DEG = -28.99181

# The Milky Way is an extended object of low surface brightness, so it needs a
# darker sky than a point source of the same total light. The default is the
# arcus visionis of a fourth-magnitude star, which is the faintest an unaided
# eye picks out and the closest available proxy for that condition.
DEFAULT_AV_DARK_DEG = 16.1


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# fixed directions carried into the frame of date
# --------------------------------------------------------------------------

def unit(ra_deg: float, dec_deg: float) -> np.ndarray:
    a, d = math.radians(ra_deg), math.radians(dec_deg)
    return np.array([math.cos(d) * math.cos(a), math.cos(d) * math.sin(a),
                     math.sin(d)])


def to_radec(v: np.ndarray):
    return (math.degrees(math.atan2(v[1], v[0])) % 360.0,
            math.degrees(math.asin(float(np.clip(v[2], -1.0, 1.0)))))


def reference_points(epochs_kyr: np.ndarray) -> pd.DataFrame:
    """Galactic centre, ecliptic pole and equinox, in the equator of date.

    The three are fixed or near-fixed in space and it is the equatorial frame
    that turns under them, so each is obtained by carrying one ICRS direction
    through the same long-term precession matrix the stars use. The equinox
    needs no direction of its own: it is the x-axis of the frame of date, so in
    ICRS it is the first row of that matrix, and in the frame of date it sits at
    right ascension and declination zero by construction.
    """
    gc0 = unit(GC_RA_ICRS_DEG, GC_DEC_ICRS_DEG)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in epochs_kyr:
            y = 1000.0 * float(t)
            rp = np.asarray(erfa.ltpb(y), dtype=float)
            ecl = np.asarray(erfa.ltpecl(y), dtype=float)
            eq = np.asarray(erfa.ltpequ(y), dtype=float)

            gc_ra, gc_dec = to_radec(rp @ gc0)
            ep_ra, ep_dec = to_radec(rp @ ecl)
            eqx_icrs = rp[0]                       # equinox of date, in ICRS
            eqx_ra, eqx_dec = to_radec(eqx_icrs)
            eps = math.degrees(math.acos(
                float(np.clip(np.dot(eq, ecl), -1.0, 1.0))))
            rows.append({"epoch_kyr": float(t), "obliquity_deg": eps,
                         "gc_ra_deg": gc_ra, "gc_dec_deg": gc_dec,
                         "ecl_pole_ra_deg": ep_ra, "ecl_pole_dec_deg": ep_dec,
                         "equinox_ra_icrs_deg": eqx_ra,
                         "equinox_dec_icrs_deg": eqx_dec})
    return pd.DataFrame(rows)


def angsep(ra1, dec1, ra2, dec2):
    """Angular separation in degrees, by the haversine form, arrays allowed."""
    a1, d1 = np.radians(ra1), np.radians(dec1)
    a2, d2 = np.radians(ra2), np.radians(dec2)
    s = (np.sin(0.5 * (d2 - d1)) ** 2
         + np.cos(d1) * np.cos(d2) * np.sin(0.5 * (a2 - a1)) ** 2)
    return np.degrees(2.0 * np.arcsin(np.sqrt(np.clip(s, 0.0, 1.0))))


def spells(epochs, mask, min_span):
    out = []
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return out
    for seg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        lo, hi = min(epochs[seg[0]], epochs[seg[-1]]), max(epochs[seg[0]],
                                                           epochs[seg[-1]])
        if hi - lo >= min_span:
            out.append((lo, hi, seg))
    return out


# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="East star, galactic centre, ecliptic pole, equinox, "
                    "and the balance of the bright sky.")
    p.add_argument("--trajectories", required=True)
    p.add_argument("--lat", type=float, default=45.5,
                   help="latitude the culmination heights are quoted at")
    p.add_argument("--lat-min", type=float, default=-60.0)
    p.add_argument("--lat-max", type=float, default=70.0)
    p.add_argument("--lat-step", type=float, default=2.0)
    p.add_argument("--vmax", type=float, default=3.0,
                   help="faintest star that can hold one of these offices")
    p.add_argument("--east-tol", type=float, default=1.0,
                   help="declination within which a star counts as due east")
    p.add_argument("--alt-min", type=float, default=60.0,
                   help="altitude above which the galactic centre counts as "
                        "overhead")
    p.add_argument("--av-dark", type=float, default=DEFAULT_AV_DARK_DEG,
                   help="solar depression the Milky Way needs, deg")
    p.add_argument("--steps", type=int, default=120,
                   help="samples of true solar longitude per year")
    p.add_argument("--min-span", type=float, default=0.5,
                   help="shortest spell reported, kyr")
    p.add_argument("--outdir", default="geometry")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log(f"Leggo {args.trajectories} ...")
    traj = read_star_table(args.trajectories,
                           ["epoch_kyr_from_year0", "HIP", "ra_deg", "dec_deg",
                            "Vmag", "NAME", "Bayer"])
    traj = normalise_epoch(traj).drop_duplicates(["HIP", "epoch"])

    lab = traj.groupby("HIP")[["NAME", "Bayer"]].first()
    labels = {}
    for hip, r in lab.iterrows():
        nm, by = str(r["NAME"]).strip(), str(r["Bayer"]).strip()
        labels[int(hip)] = (nm if nm and nm != "nan"
                            else by if by and by != "nan" else f"HIP {int(hip)}")

    ra_w = traj.pivot(index="epoch", columns="HIP", values="ra_deg").sort_index()
    de_w = traj.pivot(index="epoch", columns="HIP", values="dec_deg").sort_index()
    mg_w = traj.pivot(index="epoch", columns="HIP", values="Vmag").sort_index()
    epochs = ra_w.index.to_numpy(dtype=float)
    hips = ra_w.columns.to_numpy()
    RA = ra_w.to_numpy(dtype=float)
    DE = de_w.to_numpy(dtype=float)
    MG = mg_w.to_numpy(dtype=float)
    log(f"  {hips.size} stelle, {epochs.size} epoche "
        f"[{epochs.min():+.1f}, {epochs.max():+.1f}] kyr")

    ref = reference_points(epochs)
    ref.to_csv(outdir / "reference_points.csv", index=False,
               float_format="%.5f")
    now = int(np.argmin(np.abs(epochs - epochs.max())))

    # ======================================================================
    # 1. the east star
    # ======================================================================
    log("")
    log(f"Stella dell'est: |declinazione| < {args.east_tol:.2f}°, V < {args.vmax}")
    bright = MG <= args.vmax
    near_eq = bright & (np.abs(DE) <= args.east_tol)

    rows = []
    for i in range(epochs.size):
        k = np.flatnonzero(near_eq[i])
        if k.size == 0:
            rows.append({"epoch_kyr": float(epochs[i]), "HIP": -1, "star": "",
                         "Vmag": np.nan, "dec_deg": np.nan,
                         "n_candidates": 0})
            continue
        b = k[int(np.argmin(MG[i, k]))]           # brightest, not nearest
        rows.append({"epoch_kyr": float(epochs[i]), "HIP": int(hips[b]),
                     "star": labels[int(hips[b])], "Vmag": float(MG[i, b]),
                     "dec_deg": float(DE[i, b]), "n_candidates": int(k.size)})
    east = pd.DataFrame(rows)
    east.to_csv(outdir / "east_star.csv", index=False, float_format="%.4f")

    r0 = east.iloc[now]
    log(f"  a {epochs[now]:+.1f} kyr: {r0['star']} "
        f"(V {r0['Vmag']:.2f}, dec {r0['dec_deg']:+.3f}°), "
        f"{int(r0['n_candidates'])} candidate")
    if "Mintaka" not in str(r0["star"]) and abs(epochs[now]) < 0.5:
        log("  ATTENZIONE: all'epoca attuale dovrebbe uscire Mintaka. "
            "Controllare il file di traiettorie.")

    rows = []
    for k in range(hips.size):
        for lo, hi, seg in spells(epochs, near_eq[:, k], args.min_span):
            j = seg[int(np.argmin(np.abs(DE[seg, k])))]
            rows.append({"HIP": int(hips[k]), "star": labels[int(hips[k])],
                         "Vmag": float(np.nanmedian(MG[seg, k])),
                         "epoch_start_kyr": float(lo), "epoch_end_kyr": float(hi),
                         "duration_kyr": float(hi - lo),
                         "closest_dec_deg": float(DE[j, k]),
                         "epoch_closest_kyr": float(epochs[j])})
    esp = pd.DataFrame(rows)
    if not esp.empty:
        esp = esp.sort_values("duration_kyr", ascending=False)
        esp.to_csv(outdir / "east_star_spells.csv", index=False,
                   float_format="%.4f")
        log("  Cariche piu' lunghe:")
        for _, r in esp.head(12).iterrows():
            log(f"    {r['star']:18s} V {r['Vmag']:4.1f}  "
                f"{r['epoch_start_kyr']:+7.1f} .. {r['epoch_end_kyr']:+7.1f} kyr "
                f"({r['duration_kyr']:5.1f})")

    # ======================================================================
    # 2. the galactic centre
    # ======================================================================
    log("")
    gd = ref["gc_dec_deg"].to_numpy()
    log(f"Centro galattico: declinazione da {gd.min():+.2f}° a {gd.max():+.2f}°")
    log(f"  adesso {gd[now]:+.2f}°, cioe' "
        f"{'al minimo' if abs(gd[now] - gd.min()) < 1.0 else ''}"
        f"{'al massimo' if abs(gd[now] - gd.max()) < 1.0 else ''}"
        f"{'' if min(abs(gd[now] - gd.min()), abs(gd[now] - gd.max())) < 1.0 else 'a meta strada'}")
    alt = 90.0 - np.abs(args.lat - gd)
    log(f"  culmina a {args.lat:.1f}°: da {alt.min():.1f}° a {alt.max():.1f}° "
        f"(adesso {alt[now]:.1f}°)")
    log(f"  passa allo zenit alla latitudine = declinazione, quindi fra "
        f"{gd.min():+.1f}° e {gd.max():+.1f}°")

    lats = np.arange(args.lat_min, args.lat_max + 0.5 * args.lat_step,
                     args.lat_step)
    eps = ref["obliquity_deg"].to_numpy()
    gc_ra = np.radians(ref["gc_ra_deg"].to_numpy())
    gc_de = np.radians(gd)

    log(f"  Ore/anno con il centro sopra {args.alt_min:.0f}° e cielo buio "
        f"(Sole sotto -{args.av_dark:.1f}°) ...")
    hours = np.zeros((epochs.size, lats.size))
    season = np.zeros(epochs.size)
    for i in range(epochs.size):
        lam, days, doy = year_weights(float(epochs[i]), args.steps)
        sra, sdec = sun_position(lam, float(eps[i]))
        # The date on which the centre comes to the meridian at midnight: the
        # Sun is then diametrically opposite it in right ascension.
        season[i] = float(doy[int(np.argmin(np.abs(
            np.angle(np.exp(1j * (sra - (gc_ra[i] + math.pi)))))))])
        for j, phi in enumerate(lats):
            h_gc = half_arc(np.array([gc_de[i]]), float(phi), args.alt_min)[0]
            if h_gc <= 0.0:
                continue
            h_br = half_arc(sdec, float(phi), -args.av_dark)
            over = arc_overlap(sra - gc_ra[i], np.full(lam.size, h_gc), h_br)
            arc = np.maximum(2.0 * h_gc - over, 0.0)
            hours[i, j] = float(np.sum(arc / (2.0 * math.pi) * days) * 24.0)
        if (i + 1) % 50 == 0 or i == epochs.size - 1:
            log(f"    {i + 1}/{epochs.size} epoche")

    gcdf = pd.DataFrame({
        "epoch_kyr": epochs, "gc_ra_deg": ref["gc_ra_deg"],
        "gc_dec_deg": gd, "zenith_lat_deg": gd,
        "culmination_alt_deg_at_lat": alt,
        "midnight_culmination_doy": season,
    })
    gcdf.to_csv(outdir / "galactic_centre.csv", index=False,
                float_format="%.4f")

    rec = []
    for i in range(epochs.size):
        for j, phi in enumerate(lats):
            rec.append({"epoch_kyr": float(epochs[i]), "lat_deg": float(phi),
                        "hours_high_and_dark": hours[i, j]})
    pd.DataFrame(rec).to_csv(outdir / "galactic_zenith.csv", index=False,
                             float_format="%.3f")

    ib, jb = np.unravel_index(int(np.argmax(hours)), hours.shape)
    log("")
    log(f"  Massimo assoluto: {hours[ib, jb]:.0f} h/anno a "
        f"{lats[jb]:+.1f}° di latitudine, epoca {epochs[ib]:+.1f} kyr")
    jn = int(np.argmin(np.abs(lats - args.lat)))
    ic = int(np.argmax(hours[:, jn]))
    log(f"  A {args.lat:.1f}°: adesso {hours[now, jn]:.0f} h/anno, "
        f"massimo {hours[ic, jn]:.0f} h/anno a {epochs[ic]:+.1f} kyr "
        f"(fattore {hours[ic, jn] / max(hours[now, jn], 1e-9):.1f})")
    jz = int(np.argmin(np.abs(lats - gd[now])))
    log(f"  Alla latitudine di zenit attuale ({gd[now]:+.1f}°): "
        f"{hours[now, jz]:.0f} h/anno")

    # ======================================================================
    # 3. ecliptic pole and vernal point
    # ======================================================================
    log("")
    log("Polo dell'eclittica e punto vernale ...")
    rows = []
    for i in range(epochs.size):
        b = np.flatnonzero(MG[i] <= args.vmax)
        if b.size == 0:
            continue
        d_ep = angsep(RA[i, b], DE[i, b], ref["ecl_pole_ra_deg"].iloc[i],
                      ref["ecl_pole_dec_deg"].iloc[i])
        d_vp = angsep(RA[i, b], DE[i, b], 0.0, 0.0)
        k1, k2 = int(np.argmin(d_ep)), int(np.argmin(d_vp))
        rows.append({
            "epoch_kyr": float(epochs[i]),
            "ecl_pole_ra_deg": float(ref["ecl_pole_ra_deg"].iloc[i]),
            "ecl_pole_dec_deg": float(ref["ecl_pole_dec_deg"].iloc[i]),
            "ecl_pole_nearest": labels[int(hips[b[k1]])],
            "ecl_pole_nearest_HIP": int(hips[b[k1]]),
            "ecl_pole_nearest_Vmag": float(MG[i, b[k1]]),
            "ecl_pole_nearest_sep_deg": float(d_ep[k1]),
            "vernal_nearest": labels[int(hips[b[k2]])],
            "vernal_nearest_HIP": int(hips[b[k2]]),
            "vernal_nearest_Vmag": float(MG[i, b[k2]]),
            "vernal_nearest_sep_deg": float(d_vp[k2]),
        })
    pts = pd.DataFrame(rows)
    pts.to_csv(outdir / "ecliptic_pole_and_equinox.csv", index=False,
               float_format="%.4f")
    log(f"  Polo dell'eclittica: la stella V<{args.vmax} piu' vicina dista "
        f"da {pts['ecl_pole_nearest_sep_deg'].min():.2f}° a "
        f"{pts['ecl_pole_nearest_sep_deg'].max():.2f}°, "
        f"mediana {pts['ecl_pole_nearest_sep_deg'].median():.2f}°")
    log(f"    (per confronto, la stella polare attuale dista 0.7° dal polo "
        f"celeste: il punto davvero fisso del cielo e' molto piu' vuoto)")
    log(f"  Punto vernale: stella piu' vicina fra "
        f"{pts['vernal_nearest_sep_deg'].min():.2f}° e "
        f"{pts['vernal_nearest_sep_deg'].max():.2f}°")
    log(f"    adesso {pts.iloc[now]['vernal_nearest']} a "
        f"{pts.iloc[now]['vernal_nearest_sep_deg']:.2f}°")

    # ======================================================================
    # 4. the balance of the bright sky
    # ======================================================================
    log("")
    log("Bilancio fra i due emisferi ...")
    rows = []
    for lim in (1.0, 1.5, 2.0, 2.5, 3.0):
        for i in range(epochs.size):
            b = MG[i] <= lim
            n_n = int(np.sum(b & (DE[i] > 0.0)))
            n_s = int(np.sum(b & (DE[i] <= 0.0)))
            rows.append({"epoch_kyr": float(epochs[i]), "vmag_limit": lim,
                         "n_north": n_n, "n_south": n_s,
                         "excess_south": n_s - n_n})
    bal = pd.DataFrame(rows)
    bal.to_csv(outdir / "hemisphere_balance.csv", index=False)
    for lim in (1.5, 2.5):
        s = bal[bal["vmag_limit"] == lim]
        cur = s[np.isclose(s["epoch_kyr"], epochs[now])].iloc[0]
        log(f"  V<{lim}: adesso {int(cur['n_north'])} a nord e "
            f"{int(cur['n_south'])} a sud (eccesso sud {int(cur['excess_south'])}); "
            f"nell'arco dei dati l'eccesso va da {s['excess_south'].min()} "
            f"a {s['excess_south'].max()}")

    # ======================================================================
    # figures
    # ======================================================================
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ok = east["HIP"] > 0
    a1.scatter(east.loc[ok, "epoch_kyr"], east.loc[ok, "dec_deg"], s=6,
               c=east.loc[ok, "Vmag"], cmap="viridis_r")
    a1.axhline(0.0, color="k", lw=0.8)
    a1.set_ylabel("declinazione della stella dell'est [°]")
    a1.set_title(f"La stella che sorge a est, epoca per epoca "
                 f"(|dec| < {args.east_tol:.1f}°, V < {args.vmax})")
    a1.grid(alpha=0.3)
    if not esp.empty:
        top = esp.head(14).reset_index(drop=True)
        for y, r in top.iterrows():
            a2.plot([r["epoch_start_kyr"], r["epoch_end_kyr"]], [y, y], lw=3)
            a2.text(r["epoch_end_kyr"], y, f"  {r['star']}", fontsize=7,
                    va="center")
        a2.set_yticks([])
    a2.set_xlabel("epoca [kyr dall'anno 0]")
    a2.set_title("durata delle cariche piu' lunghe")
    fig.tight_layout()
    fig.savefig(outdir / "east_star.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    a1.plot(epochs, gd, lw=1.4, color="darkorange",
            label="declinazione del centro galattico")
    a1.plot(epochs, alt, lw=1.2, color="navy",
            label=f"altezza in culminazione a {args.lat:.1f}°")
    a1.axhline(0.0, color="k", lw=0.6)
    a1.axvline(epochs[now], color="0.6", lw=0.8, ls=":")
    a1.set_ylabel("gradi")
    a1.set_title("Il centro galattico sale e scende con la precessione\n"
                 "la sua latitudine eclittica e' fissa, quindi la "
                 "declinazione oscilla di due volte l'obliquita'")
    a1.legend(fontsize=8)
    a1.grid(alpha=0.3)

    im = a2.imshow(hours.T, origin="lower", aspect="auto",
                   extent=[epochs[0], epochs[-1], lats[0], lats[-1]],
                   cmap="magma")
    fig.colorbar(im, ax=a2, label="ore/anno")
    a2.plot(epochs, gd, color="c", lw=1.0, ls="--",
            label="latitudine di passaggio allo zenit")
    a2.legend(fontsize=8, loc="upper right")
    a2.set_xlabel("epoca [kyr dall'anno 0]")
    a2.set_ylabel("latitudine [°]")
    a2.set_title(f"Ore all'anno con il centro galattico sopra "
                 f"{args.alt_min:.0f}° e cielo buio")
    fig.tight_layout()
    fig.savefig(outdir / "galactic_centre.png", dpi=args.dpi,
                bbox_inches="tight")
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    a1.plot(pts["epoch_kyr"], pts["ecl_pole_nearest_sep_deg"], lw=1.2,
            color="purple", label="polo dell'eclittica")
    a1.plot(pts["epoch_kyr"], pts["vernal_nearest_sep_deg"], lw=1.2,
            color="teal", label="punto vernale")
    a1.axhline(0.74, color="r", lw=0.8, ls="--",
               label="distanza attuale Polare-polo celeste")
    a1.set_ylabel(f"distanza della stella V<{args.vmax} piu' vicina [°]")
    a1.set_title("Quanto sono marcati i due punti notevoli\n"
                 "il polo dell'eclittica non si muove e resta vuoto; "
                 "il punto vernale gira e incontra tutto")
    a1.legend(fontsize=8)
    a1.grid(alpha=0.3)

    for lim, col in ((1.5, "firebrick"), (2.5, "steelblue")):
        s = bal[bal["vmag_limit"] == lim]
        a2.plot(s["epoch_kyr"], s["excess_south"], lw=1.2, color=col,
                label=f"V < {lim}")
    a2.axhline(0.0, color="k", lw=0.8)
    a2.set_xlabel("epoca [kyr dall'anno 0]")
    a2.set_ylabel("stelle in piu' a sud")
    a2.set_title("Squilibrio fra i due emisferi: chi migra verso sud "
                 "guadagna cielo")
    a2.legend(fontsize=8)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "points_and_balance.png", dpi=args.dpi,
                bbox_inches="tight")
    plt.close(fig)

    log("")
    log(f"Scritti {outdir}/east_star.csv, east_star_spells.csv, "
        f"galactic_centre.csv, galactic_zenith.csv, "
        f"ecliptic_pole_and_equinox.csv, hemisphere_balance.csv, "
        f"reference_points.csv, east_star.png, galactic_centre.png, "
        f"points_and_balance.png")


if __name__ == "__main__":
    main()
