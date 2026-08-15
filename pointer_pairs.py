#!/usr/bin/env python3
"""
pointer_pairs.py

Chains of stars whose line, extended, finds the celestial pole.

Merak and Dubhe in the Plough are the known case: the line from one to the other,
carried on about five times its own length, arrives at Polaris. The technique
matters because it needs no pole star, only two or more conspicuous stars in the
right relation, and it therefore works through the long stretches -- most of the
last two hundred millennia -- when nothing bright sits near the pole at all.
Where pole_stars.py asks what marker the sky offered, this asks what construction
it offered instead.

Chains of two to five stars are searched, not pairs alone. Three or four stars in
a row fix a direction far better than two, because the eye judges a line through
several points more surely than the prolongation of a single segment, and a
misplaced member shows up at once.

Geometry
--------
The stars of a chain must lie on one great circle, and that circle must pass near
the pole. Four numbers describe how well the thing serves.

    miss        perpendicular distance from the pole to the fitted great circle.
                Follow the line and this is how far to one side of true north
                you end up: the same quantity as, and directly comparable to,
                the pole distance of a single marker star.

    scatter     how far the members sit off their own line. A chain that is not
                straight is not a line anybody would follow.

    gap         the largest step between consecutive members. Left free it
                selects pairs twenty degrees apart, and a prolongation across
                that much empty sky is a guess, not a sighting.

    ratio       the throw beyond the last star, in units of the chain's own
                length -- the "five times" of the rule, which has to be judged
                by eye.

Miss and ratio are kept apart rather than combined, because they behave
differently and no weighting between them would be defensible: the miss puts you
off across the line whatever you do, while misjudging the throw puts you off
along it by that fraction of it, so a long throw amplifies a proportional error
and a short one does not.

Search
------
Every chain that points at the pole ends in a pair that points at the pole: its
last two members lie on the same great circle with the pole beyond them. So the
qualifying pairs are found first and grown backwards, which avoids enumerating
the hundreds of millions of five-star combinations and misses nothing.

Both poles are done, and the known constructions -- Merak and Dubhe in the north,
the long axis of the Cross in the south -- are evaluated at the present epoch and
printed, so that a wrong sign or a reversed pair cannot pass unnoticed.

Usage
-----
    python3 pointer_pairs.py --trajectories BSGRID/trajectories.csv --outdir pointers
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

REFERENCE_PAIRS = {
    "nord": (53910, 54061, "Merak -> Dubhe -> Polaris"),
    "sud": (61084, 60718, "Gacrux -> Acrux, asse della Croce"),
}


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


def unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    cd = np.cos(dec)
    return np.column_stack((cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)))


def angle_between(u: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(np.sum(u * w, axis=-1), -1.0, 1.0)))


def wrap180(x):
    return (x + 180.0) % 360.0 - 180.0


def fit_great_circle(pts: np.ndarray):
    """Normal of the great circle best fitting a set of directions.

    The circle through a set of unit vectors is the plane through the origin
    closest to them, whose normal is the least singular vector. The smallest
    singular value measures how far they sit off it, which is the scatter that
    says whether they form a line at all.
    """
    u, s, vt = np.linalg.svd(pts - 0.0, full_matrices=False)
    n = vt[-1]
    n = n / np.linalg.norm(n)
    off = np.degrees(np.arcsin(np.clip(np.abs(pts @ n), 0.0, 1.0)))
    return n, float(np.sqrt(np.mean(off ** 2)))


def pair_quality(v, ia, ib, pole):
    """Miss, separation, throw and ratio for every ordered pair at one epoch."""
    a, b = v[ia], v[ib]
    n = np.cross(a, b)
    ln = np.linalg.norm(n, axis=1)
    good = ln > 1e-12
    n = np.where(good[:, None], n / np.where(good[:, None], ln[:, None], 1.0), 0.0)

    dot_np = n @ pole
    miss = np.degrees(np.arcsin(np.clip(np.abs(dot_np), 0.0, 1.0)))

    pp = pole[None, :] - dot_np[:, None] * n
    lp = np.linalg.norm(pp, axis=1)
    ok = good & (lp > 1e-12)
    pp = np.where(ok[:, None], pp / np.where(ok[:, None], lp[:, None], 1.0), 0.0)

    d_ab = angle_between(a, b)
    d_bp = angle_between(b, pp)
    d_ap = angle_between(a, pp)
    forward = ok & (np.abs(d_ap - (d_ab + d_bp)) < 0.01)
    ratio = np.where(d_ab > 1e-9, d_bp / np.where(d_ab > 1e-9, d_ab, 1.0), np.inf)
    return miss, d_ab, d_bp, ratio, forward


def grow_chain(v, ka, kb, pole, gap_max, collinear_tol, max_stars):
    """Extend a qualifying terminal pair backwards along its own great circle.

    Members are ordered by arc distance from the pole's foot on the circle, all
    on the side the pair lies on, and the chain is taken as far back as
    consecutive steps stay within the gap. The circle is then refitted to
    everything kept, so that the miss reported belongs to the chain rather than
    to the two stars it started from.
    """
    a, b = v[ka], v[kb]
    n = np.cross(a, b)
    ln = np.linalg.norm(n)
    if ln < 1e-12:
        return None
    n = n / ln

    off = np.degrees(np.arcsin(np.clip(np.abs(v @ n), 0.0, 1.0)))
    near = np.flatnonzero(off <= collinear_tol)
    if near.size < 2:
        return None

    e1 = a - (a @ n) * n
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n, e1)

    pp = pole - (pole @ n) * n
    if np.linalg.norm(pp) < 1e-12:
        return None
    pp = pp / np.linalg.norm(pp)

    t = np.degrees(np.arctan2(v[near] @ e2, v[near] @ e1))
    t_p = np.degrees(np.arctan2(pp @ e2, pp @ e1))
    s = wrap180(t - t_p)                       # signed arc from the pole's foot

    side = np.sign(wrap180(np.degrees(np.arctan2(b @ e2, b @ e1)) - t_p))
    keep = np.flatnonzero(np.sign(s) == side)
    if keep.size < 2:
        return None
    idx = near[keep]
    dist = np.abs(s[keep])
    order = np.argsort(dist)
    idx, dist = idx[order], dist[order]

    chain = [int(idx[0])]
    for j in range(1, idx.size):
        if len(chain) >= max_stars:
            break
        if dist[j] - dist[j - 1] <= gap_max:
            chain.append(int(idx[j]))
        else:
            break
    if len(chain) < 2:
        return None

    pts = v[chain]
    n_fit, scatter = fit_great_circle(pts)
    miss = float(np.degrees(np.arcsin(np.clip(abs(float(n_fit @ pole)), 0.0, 1.0))))
    span = float(dist[len(chain) - 1] - dist[0])
    gaps = np.diff(dist[:len(chain)])
    return {
        "members": chain,                       # first is nearest the pole
        "miss_deg": miss,
        "scatter_deg": scatter,
        "span_deg": span,
        "max_gap_deg": float(gaps.max()) if gaps.size else 0.0,
        "extension_deg": float(dist[0]),
        "ratio": float(dist[0] / span) if span > 1e-9 else np.inf,
    }


def spells(epochs, mask, min_span):
    out = []
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return out
    for seg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        lo = min(epochs[seg[0]], epochs[seg[-1]])
        hi = max(epochs[seg[0]], epochs[seg[-1]])
        if hi - lo >= min_span:
            out.append((lo, hi, seg))
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Chains of stars whose line, extended, finds the pole.")
    p.add_argument("--trajectories", required=True)
    p.add_argument("--outdir", default="pointers")
    p.add_argument("--vmax", type=float, default=3.0,
                   help="every member must be at least this bright")
    p.add_argument("--gap-min", type=float, default=2.0,
                   help="closest two consecutive members may be, degrees: below "
                        "this the direction of the line cannot be judged")
    p.add_argument("--gap-max", type=float, default=10.0,
                   help="furthest apart two consecutive members may be, degrees. "
                        "A prolongation across more empty sky than this is a "
                        "guess rather than a sighting")
    p.add_argument("--max-stars", type=int, default=5,
                   help="longest chain searched")
    p.add_argument("--min-stars", type=int, default=2,
                   help="shortest chain kept; 2 is the plain pointer pair")
    p.add_argument("--collinear-tol", type=float, default=2.0,
                   help="how far off the line a member may sit, degrees")
    p.add_argument("--miss-max", type=float, default=3.0,
                   help="largest tolerated distance of the pole from the line")
    p.add_argument("--ratio-min", type=float, default=1.0)
    p.add_argument("--ratio-max", type=float, default=8.0,
                   help="longest throw kept, in units of the chain's own length")
    p.add_argument("--epoch-step", type=float, default=0.5)
    p.add_argument("--min-span", type=float, default=1.0,
                   help="shortest spell of a chain reported, kyr")
    p.add_argument("--top", type=int, default=5,
                   help="chains kept per epoch and pole")
    p.add_argument("--max-seeds", type=int, default=300,
                   help="qualifying pairs grown per epoch, best first")
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
    epochs_all = ra_w.index.to_numpy(dtype=float)
    hips = ra_w.columns.to_numpy()
    RA, DE, MG = ra_w.to_numpy(float), de_w.to_numpy(float), mg_w.to_numpy(float)

    stride = 1
    if epochs_all.size > 1:
        stride = max(1, int(round(args.epoch_step / np.min(np.diff(epochs_all)))))
    sel = np.arange(0, epochs_all.size, stride)
    epochs = epochs_all[sel]
    log(f"  {hips.size} stelle, {epochs.size} epoche "
        f"[{epochs.min():+.1f}, {epochs.max():+.1f}] kyr")
    log(f"  catene di {args.min_stars}-{args.max_stars} stelle | passo fra membri "
        f"{args.gap_min}-{args.gap_max}° | fuori linea max {args.collinear_tol}°")

    poles = {"nord": np.array([0.0, 0.0, 1.0]),
             "sud": np.array([0.0, 0.0, -1.0])}

    rows = []
    seen: dict[tuple, list[int]] = {}

    for t_i, i in enumerate(sel):
        bright = np.flatnonzero(MG[i] <= args.vmax)
        if bright.size < 2:
            continue
        v = unit_vectors(RA[i, bright], DE[i, bright])

        ia, ib = np.meshgrid(np.arange(bright.size), np.arange(bright.size),
                             indexing="ij")
        ia, ib = ia.ravel(), ib.ravel()
        m = ia != ib
        ia, ib = ia[m], ib[m]

        for pole_name, pvec in poles.items():
            miss, d_ab, d_bp, ratio, fwd = pair_quality(v, ia, ib, pvec)
            ok = (fwd & (d_ab >= args.gap_min) & (d_ab <= args.gap_max)
                  & (miss <= args.miss_max))
            seeds = np.flatnonzero(ok)
            if seeds.size == 0:
                continue
            seeds = seeds[np.argsort(miss[seeds])][:args.max_seeds]

            found = {}
            for k in seeds:
                ch = grow_chain(v, int(ia[k]), int(ib[k]), pvec,
                                args.gap_max, args.collinear_tol, args.max_stars)
                if ch is None or len(ch["members"]) < args.min_stars:
                    continue
                if not (args.ratio_min <= ch["ratio"] <= args.ratio_max):
                    continue
                if ch["miss_deg"] > args.miss_max:
                    continue
                if ch["max_gap_deg"] > args.gap_max:
                    continue
                key = tuple(sorted(int(hips[bright[j]]) for j in ch["members"]))
                prev = found.get(key)
                if prev is None or ch["miss_deg"] < prev["miss_deg"]:
                    found[key] = ch

            ranked = sorted(found.items(),
                            key=lambda kv: (-len(kv[1]["members"]),
                                            kv[1]["miss_deg"]))
            for rank, (key, ch) in enumerate(ranked[:args.top], 1):
                hs = [int(hips[bright[j]]) for j in ch["members"]]
                rows.append({
                    "epoch_kyr": epochs[t_i], "pole": pole_name, "rank": rank,
                    "n_stars": len(hs),
                    "stars": " -> ".join(labels.get(h, str(h))
                                         for h in reversed(hs)),
                    "HIPs": ",".join(str(h) for h in reversed(hs)),
                    "miss_deg": ch["miss_deg"], "scatter_deg": ch["scatter_deg"],
                    "span_deg": ch["span_deg"], "max_gap_deg": ch["max_gap_deg"],
                    "extension_deg": ch["extension_deg"], "ratio": ch["ratio"],
                    "Vmag_max": float(max(MG[i, bright[j]]
                                          for j in ch["members"])),
                })
                seen.setdefault((pole_name, key), []).append(t_i)

    if not rows:
        raise SystemExit("Nessuna catena soddisfa i criteri.")

    ch_df = pd.DataFrame(rows)
    ch_df.to_csv(outdir / "chains.csv", index=False, float_format="%.4f")

    # ---- validation on the known constructions ---------------------------
    log("")
    log("  Verifica sulle costruzioni note, all'epoca piu' recente:")
    i_now = sel[-1]
    for pole_name, (ha, hb, descr) in REFERENCE_PAIRS.items():
        if ha not in hips or hb not in hips:
            log(f"    {descr}: stelle non nel catalogo")
            continue
        ka = int(np.flatnonzero(hips == ha)[0])
        kb = int(np.flatnonzero(hips == hb)[0])
        v2 = unit_vectors(RA[i_now, [ka, kb]], DE[i_now, [ka, kb]])
        miss, d_ab, d_bp, ratio, fwd = pair_quality(
            v2, np.array([0]), np.array([1]), poles[pole_name])
        log(f"    {descr}")
        log(f"      separazione {d_ab[0]:5.2f}° | estensione {d_bp[0]:6.2f}° | "
            f"rapporto {ratio[0]:5.2f} | scarto {miss[0]:5.2f}° | "
            f"{'in avanti' if fwd[0] else 'ALL INDIETRO'}")

    # ---- how many stars the best chain has, through time ------------------
    log("")
    log("  Catene trovate per numero di stelle:")
    for k in sorted(ch_df["n_stars"].unique()):
        s = ch_df[ch_df.n_stars == k]
        log(f"    {k} stelle: {len(s):5d} occorrenze | scarto mediano "
            f"{s['miss_deg'].median():4.2f}° | rapporto mediano "
            f"{s['ratio'].median():4.1f}")

    # ---- spells -----------------------------------------------------------
    srows = []
    for (pole_name, key), tlist in seen.items():
        m = np.zeros(epochs.size, dtype=bool)
        m[np.array(tlist)] = True
        for lo, hi, seg in spells(epochs, m, args.min_span):
            sub = ch_df[(ch_df.pole == pole_name)
                        & (ch_df.HIPs.str.split(",").apply(
                            lambda x: tuple(sorted(int(i) for i in x)) == key))
                        & (ch_df.epoch_kyr >= lo) & (ch_df.epoch_kyr <= hi)]
            srows.append({
                "pole": pole_name, "n_stars": len(key),
                "stars": sub["stars"].iloc[0] if not sub.empty else "",
                "HIPs": ",".join(str(h) for h in key),
                "epoch_start_kyr": float(lo), "epoch_end_kyr": float(hi),
                "duration_kyr": float(hi - lo),
                "best_miss_deg": float(sub["miss_deg"].min()) if not sub.empty else np.nan,
                "median_ratio": float(sub["ratio"].median()) if not sub.empty else np.nan,
            })
    sp = pd.DataFrame(srows).sort_values(
        ["pole", "n_stars", "duration_kyr"], ascending=[True, False, False])
    sp.to_csv(outdir / "spells.csv", index=False, float_format="%.4f")

    log("")
    log("  Catene piu' durature, per polo e lunghezza:")
    for pole_name in ("nord", "sud"):
        s = sp[sp.pole == pole_name]
        if s.empty:
            log(f"    {pole_name}: nessuna")
            continue
        log(f"    polo {pole_name}:")
        for k in sorted(s["n_stars"].unique(), reverse=True):
            for _, r in s[s.n_stars == k].head(4).iterrows():
                log(f"      [{k}] {str(r['stars'])[:46]:46s} "
                    f"{r['epoch_start_kyr']:+7.1f}..{r['epoch_end_kyr']:+7.1f} "
                    f"({r['duration_kyr']:5.1f}) scarto {r['best_miss_deg']:4.2f}°")

    # ---- figures ----------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, pole_name in zip(axes, ("nord", "sud")):
        s = ch_df[(ch_df.pole == pole_name) & (ch_df["rank"] == 1)]
        s = s.sort_values("epoch_kyr")
        if not s.empty:
            ax.plot(s["epoch_kyr"], s["miss_deg"], lw=1.2, color="navy",
                    label="scarto della migliore catena")
            ax2 = ax.twinx()
            ax2.step(s["epoch_kyr"], s["n_stars"], where="mid", lw=1.0,
                     color="darkorange", alpha=0.8)
            ax2.set_ylabel("stelle nella catena", color="darkorange", fontsize=9)
            ax2.tick_params(axis="y", colors="darkorange")
            ax2.set_ylim(1.5, args.max_stars + 0.5)
        ax.set_ylabel(f"scarto dal polo {pole_name} [°]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    axes[1].set_xlabel("Epoca [kyr dall'anno 0]")
    axes[0].set_title("Migliore allineamento che indica il polo\n"
                      "scarto = errore di direzione; in arancione quante stelle "
                      "lo compongono")
    fig.tight_layout()
    fig.savefig(outdir / "best_chain.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    for ax, pole_name in zip(axes, ("nord", "sud")):
        s = sp[sp.pole == pole_name].sort_values(
            ["n_stars", "duration_kyr"], ascending=[False, False]).head(20)
        if s.empty:
            ax.axis("off")
            continue
        for y, (_, r) in enumerate(s.iterrows()):
            ax.barh(y, r["epoch_end_kyr"] - r["epoch_start_kyr"],
                    left=r["epoch_start_kyr"], height=0.62,
                    color=plt.cm.viridis_r(min(r["best_miss_deg"] / args.miss_max, 1)))
            ax.text(r["epoch_end_kyr"], y, f" {r['best_miss_deg']:.1f}°",
                    va="center", fontsize=7)
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels([f"[{r['n_stars']}] {str(r['stars'])[:42]}"
                            for _, r in s.iterrows()], fontsize=6.5)
        ax.invert_yaxis()
        ax.set_ylabel(f"polo {pole_name}")
        ax.grid(axis="x", alpha=0.3)
    axes[1].set_xlabel("Epoca [kyr dall'anno 0]")
    axes[0].set_title("Quando ciascun allineamento indicava il polo\n"
                      "fra parentesi il numero di stelle; colore ed etichetta: "
                      "scarto minimo")
    fig.tight_layout()
    fig.savefig(outdir / "spells.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    log("")
    log(f"Scritti {outdir}/chains.csv, spells.csv, best_chain.png, spells.png")


if __name__ == "__main__":
    main()
