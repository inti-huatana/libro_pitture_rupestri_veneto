#!/usr/bin/env python3
"""
pointer_pairs.py

Pairs of stars whose line, extended, finds the celestial pole.

Merak and Dubhe in the Plough are the known case: the line from one to the
other, carried on about five times its own length, arrives at Polaris. The
technique matters because it does not need a pole star. It needs only two
conspicuous stars in the right relation, and it therefore works through the long
stretches -- most of the last two hundred millennia -- when nothing bright sits
near the pole at all. Where pole_stars.py asks what marker the sky offered, this
asks what construction it offered instead.

Geometry
--------
Two stars define a great circle. Three numbers describe how well it serves.

    miss        the perpendicular distance from the pole to that great circle.
                Follow the line and this is how far to one side of true north
                you end up, in degrees: the same quantity, and directly
                comparable to, the pole distance of a single marker star.

    extension   how far beyond the second star the pole lies, in degrees.

    ratio       that distance in units of the pair's own separation, the "five
                times" of the rule. It is what has to be remembered and judged
                by eye.

The two errors are kept apart rather than combined, because they behave
differently and no weighting between them would be defensible. The miss puts you
off across the line whatever you do. Misjudging the extension puts you off along
it, by roughly that fraction of the extension, so a long throw amplifies a
proportional misjudgement and a short one does not: at a ratio of five, being
twenty per cent out on the length costs more than a typical miss.

The pole must lie beyond the second star, not behind the first, which is checked
by requiring the three points to fall in that order along the circle.

Both poles are done. The southern sky has its own known constructions -- the long
axis of the Cross, the Pointers of Centaurus -- and they serve as the check that
the calculation is right, together with Merak and Dubhe in the north.

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

# Known constructions, checked at the present epoch so that a wrong sign or a
# reversed pair cannot pass unnoticed.
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


def evaluate_pairs(v: np.ndarray, ia: np.ndarray, ib: np.ndarray,
                   pole: np.ndarray):
    """Miss, extension and ratio for every ordered pair, at one epoch.

    The great circle through the pair has unit normal a x b, so the pole's
    perpendicular distance from it is the arcsine of the normal's component
    along the pole. Projecting the pole into the plane of the circle gives the
    point on the line nearest to it, and the three arc lengths then say whether
    the pole lies beyond the second star or behind the first.
    """
    a, b = v[ia], v[ib]
    n = np.cross(a, b)
    ln = np.linalg.norm(n, axis=1)
    good = ln > 1e-12
    n = np.where(good[:, None], n / np.where(good[:, None], ln[:, None], 1.0), 0.0)

    dot_np = n @ pole
    miss = np.degrees(np.arcsin(np.clip(np.abs(dot_np), 0.0, 1.0)))

    # Pole projected onto the plane of the great circle.
    pp = pole[None, :] - dot_np[:, None] * n
    lp = np.linalg.norm(pp, axis=1)
    ok = good & (lp > 1e-12)
    pp = np.where(ok[:, None], pp / np.where(ok[:, None], lp[:, None], 1.0), 0.0)

    d_ab = angle_between(a, b)
    d_bp = angle_between(b, pp)
    d_ap = angle_between(a, pp)

    # b between a and the pole's foot: the line points forward, not backward.
    forward = ok & (np.abs(d_ap - (d_ab + d_bp)) < 0.01)
    ratio = np.where(d_ab > 1e-9, d_bp / np.where(d_ab > 1e-9, d_ab, 1.0), np.inf)
    return miss, d_ab, d_bp, ratio, forward


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
        description="Star pairs whose line, extended, finds the pole.")
    p.add_argument("--trajectories", required=True)
    p.add_argument("--outdir", default="pointers")
    p.add_argument("--vmax", type=float, default=3.0,
                   help="both stars must be at least this bright; a pointer "
                        "nobody can pick out is no pointer")
    p.add_argument("--sep-min", type=float, default=2.0,
                   help="closest the two stars may be, degrees: below this the "
                        "direction of the line cannot be judged")
    p.add_argument("--sep-max", type=float, default=25.0,
                   help="furthest apart they may be, degrees")
    p.add_argument("--miss-max", type=float, default=3.0,
                   help="largest tolerated perpendicular distance of the pole "
                        "from the line, degrees")
    p.add_argument("--ratio-min", type=float, default=1.0)
    p.add_argument("--ratio-max", type=float, default=8.0,
                   help="longest throw kept, in units of the pair separation; "
                        "beyond this a proportional misjudgement swamps the miss")
    p.add_argument("--epoch-step", type=float, default=0.5,
                   help="subsample the file's epochs to this spacing, kyr")
    p.add_argument("--min-span", type=float, default=1.0,
                   help="shortest spell of a pair reported, kyr")
    p.add_argument("--top", type=int, default=5,
                   help="pairs kept per epoch and pole")
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
    RA, DE, MG = (ra_w.to_numpy(float), de_w.to_numpy(float), mg_w.to_numpy(float))

    stride = 1
    if epochs_all.size > 1:
        base = np.min(np.diff(epochs_all))
        stride = max(1, int(round(args.epoch_step / base)))
    sel = np.arange(0, epochs_all.size, stride)
    epochs = epochs_all[sel]
    log(f"  {hips.size} stelle, {epochs.size} epoche "
        f"[{epochs.min():+.1f}, {epochs.max():+.1f}] kyr")

    poles = {"nord": np.array([0.0, 0.0, 1.0]),
             "sud": np.array([0.0, 0.0, -1.0])}

    best_rows, all_rows = [], []
    # Membership of each pair, epoch by epoch, for the spells below.
    seen: dict[tuple[str, int, int], list[int]] = {}

    for t_i, i in enumerate(sel):
        bright = np.flatnonzero(MG[i] <= args.vmax)
        if bright.size < 2:
            continue
        v = unit_vectors(RA[i, bright], DE[i, bright])

        ia, ib = np.meshgrid(np.arange(bright.size), np.arange(bright.size),
                             indexing="ij")
        ia, ib = ia.ravel(), ib.ravel()
        keep = ia != ib
        ia, ib = ia[keep], ib[keep]

        for pole_name, pvec in poles.items():
            miss, d_ab, d_bp, ratio, fwd = evaluate_pairs(v, ia, ib, pvec)
            ok = (fwd & (d_ab >= args.sep_min) & (d_ab <= args.sep_max)
                  & (miss <= args.miss_max)
                  & (ratio >= args.ratio_min) & (ratio <= args.ratio_max))
            idx = np.flatnonzero(ok)
            if idx.size == 0:
                continue
            order = idx[np.argsort(miss[idx])]
            for rank, k in enumerate(order[:args.top], 1):
                ha, hb = int(hips[bright[ia[k]]]), int(hips[bright[ib[k]]])
                row = {
                    "epoch_kyr": epochs[t_i], "pole": pole_name, "rank": rank,
                    "HIP_a": ha, "HIP_b": hb,
                    "star_a": labels.get(ha, ""), "star_b": labels.get(hb, ""),
                    "Vmag_a": float(MG[i, bright[ia[k]]]),
                    "Vmag_b": float(MG[i, bright[ib[k]]]),
                    "separation_deg": float(d_ab[k]),
                    "extension_deg": float(d_bp[k]),
                    "ratio": float(ratio[k]),
                    "miss_deg": float(miss[k]),
                }
                best_rows.append(row)
                if rank == 1:
                    all_rows.append(row)
            for k in order:
                key = (pole_name, int(hips[bright[ia[k]]]),
                       int(hips[bright[ib[k]]]))
                seen.setdefault(key, []).append(t_i)

    if not best_rows:
        raise SystemExit("Nessuna coppia soddisfa i criteri.")

    bp = pd.DataFrame(best_rows)
    bp.to_csv(outdir / "pairs.csv", index=False, float_format="%.4f")

    # ---- validation against the known constructions ----------------------
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
        miss, d_ab, d_bp, ratio, fwd = evaluate_pairs(
            v2, np.array([0]), np.array([1]), poles[pole_name])
        log(f"    {descr}")
        log(f"      separazione {d_ab[0]:5.2f}° | estensione {d_bp[0]:6.2f}° | "
            f"rapporto {ratio[0]:5.2f} | scarto {miss[0]:5.2f}° | "
            f"{'in avanti' if fwd[0] else 'ALL INDIETRO'}")

    # ---- spells ----------------------------------------------------------
    rows = []
    for (pole_name, ha, hb), tlist in seen.items():
        m = np.zeros(epochs.size, dtype=bool)
        m[np.array(tlist)] = True
        for lo, hi, seg in spells(epochs, m, args.min_span):
            sub = bp[(bp.pole == pole_name) & (bp.HIP_a == ha) & (bp.HIP_b == hb)
                     & (bp.epoch_kyr >= lo) & (bp.epoch_kyr <= hi)]
            rows.append({
                "pole": pole_name, "HIP_a": ha, "HIP_b": hb,
                "star_a": labels.get(ha, ""), "star_b": labels.get(hb, ""),
                "epoch_start_kyr": float(lo), "epoch_end_kyr": float(hi),
                "duration_kyr": float(hi - lo),
                "best_miss_deg": float(sub["miss_deg"].min()) if not sub.empty else np.nan,
                "median_ratio": float(sub["ratio"].median()) if not sub.empty else np.nan,
            })
    sp = pd.DataFrame(rows).sort_values(["pole", "duration_kyr"],
                                        ascending=[True, False])
    sp.to_csv(outdir / "spells.csv", index=False, float_format="%.4f")

    log("")
    log("  Coppie piu' durature, per polo:")
    for pole_name in ("nord", "sud"):
        s = sp[sp.pole == pole_name].head(8)
        if s.empty:
            log(f"    {pole_name}: nessuna")
            continue
        log(f"    polo {pole_name}:")
        for _, r in s.iterrows():
            log(f"      {r['star_a']:14s} -> {r['star_b']:14s} | "
                f"{r['epoch_start_kyr']:+7.1f} .. {r['epoch_end_kyr']:+7.1f} kyr "
                f"({r['duration_kyr']:5.1f}) | scarto min {r['best_miss_deg']:4.2f}° "
                f"| rapporto {r['median_ratio']:4.1f}")

    # ---- figures ---------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, pole_name in zip(axes, ("nord", "sud")):
        s = bp[(bp.pole == pole_name) & (bp["rank"] == 1)].sort_values("epoch_kyr")
        if not s.empty:
            ax.plot(s["epoch_kyr"], s["miss_deg"], lw=1.3, color="navy",
                    label="scarto della migliore coppia")
            ax2 = ax.twinx()
            ax2.plot(s["epoch_kyr"], s["ratio"], lw=1.0, color="darkorange",
                     alpha=0.7, label="rapporto di estensione")
            ax2.set_ylabel("rapporto", color="darkorange", fontsize=9)
            ax2.tick_params(axis="y", colors="darkorange")
        ax.set_ylabel(f"scarto dal polo {pole_name} [°]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    axes[1].set_xlabel("Epoca [kyr dall'anno 0]")
    axes[0].set_title("Migliore coppia che indica il polo: lo scarto e' l'errore "
                      "di direzione\nil rapporto e' quante volte va estesa la "
                      "congiungente, e amplifica l'errore di stima")
    fig.tight_layout()
    fig.savefig(outdir / "best_pair.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    for ax, pole_name in zip(axes, ("nord", "sud")):
        s = sp[sp.pole == pole_name].head(18)
        if s.empty:
            ax.axis("off")
            continue
        names = [f"{r['star_a']} → {r['star_b']}" for _, r in s.iterrows()]
        for y, (_, r) in enumerate(s.iterrows()):
            ax.barh(y, r["epoch_end_kyr"] - r["epoch_start_kyr"],
                    left=r["epoch_start_kyr"], height=0.6,
                    color=plt.cm.viridis_r(min(r["best_miss_deg"] / args.miss_max, 1)))
            ax.text(r["epoch_end_kyr"], y, f" {r['best_miss_deg']:.1f}°",
                    va="center", fontsize=7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.invert_yaxis()
        ax.set_ylabel(f"polo {pole_name}")
        ax.grid(axis="x", alpha=0.3)
    axes[1].set_xlabel("Epoca [kyr dall'anno 0]")
    axes[0].set_title("Quando ciascuna coppia indicava il polo\n"
                      "(colore ed etichetta: scarto minimo raggiunto)")
    fig.tight_layout()
    fig.savefig(outdir / "spells.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    log("")
    log(f"Scritti {outdir}/pairs.csv, spells.csv, best_pair.png, spells.png")


if __name__ == "__main__":
    main()
