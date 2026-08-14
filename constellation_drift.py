#!/usr/bin/env python3
"""
constellation_drift.py

How much each constellation's shape deforms under stellar proper motion.

The deformation is a property of the stars, not of the lines drawn between them:
a constellation only selects which pairs of stars to look at, and the angular
separation of a pair changes by the same amount whoever grouped it with whatever
else. So the figures used here are only a presentation layer, and the modern IAU
ones serve as well as any -- better, in fact, since they are uniform in
construction and carry none of the reconstruction bias of an ethnographic
source. Because --culture takes any sky culture the parser produces, that claim
can be checked rather than assumed: run the same analysis on a second culture
and see whether the ranking moves.

Measurement
-----------
For a figure of n stars, all n(n-1)/2 internal angular separations are computed
at every epoch and divided by their values at the reference epoch. Internal
separations are invariant under precession, which is a rotation of the whole
frame, so what is left is proper motion alone.

Three numbers come out of the ratios, kept apart because they mean different
things:

    scale       their median. A group receding or approaching shrinks or grows
                as a whole without changing shape, and for a bound cluster this
                is nearly all of the effect.

    total       the RMS relative change of the separations, scale included: the
                whole alteration of the pattern. For a two-star figure it is all
                there is, which is why the shape measure alone cannot decide
                whether a figure is worth keeping.

    distortion  the RMS spread of the ratios about the median, in per cent: the
                part no rescaling can undo. A few per cent is imperceptible, a
                few tens of per cent is a different pattern. Identically zero
                for a pair, a single separation being its own median.

Which star is responsible is found by leaving each one out in turn and taking
the one whose absence most reduces the distortion. What governs it is proper
motion, which grows with nearness but also with the star's own space velocity,
so the nearest member is the usual culprit without being the inevitable one: the
Southern Cross comes apart because of Gacrux at 27 pc, whose removal takes the
figure from 103 per cent distortion to 6 while its other three stars all lie
beyond 85 pc and hold their positions relative to one another, but Cassiopeia is
dismantled by Schedar at 71 pc rather than by Ruchbah at 31. The correlation
between nearness and deformation is therefore computed across the whole set and
printed, instead of being asserted.

No threshold of recognisability is imposed. The epoch at which the distortion
first crosses 5, 10 and 25 per cent is reported, and the reader picks.

Usage
-----
    python3 constellation_drift.py \
        --trajectories fullcat.csv.gz \
        --skycultures skycultures_csv \
        --culture modern_iau \
        --outdir drift
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

# Levels at which the crossing epoch is reported, in per cent of shape change.
CROSS_LEVELS = (5.0, 10.0, 25.0)

# The sky cultures spell the IAU figures out in English. Astronomy names them in
# Latin, so the English is translated back rather than shown.
LATIN_NAMES: dict[str, str] = {
    "Andromeda": "Andromeda", "Air Pump": "Antlia", "Bird of Paradise": "Apus",
    "Water Bearer": "Aquarius", "Eagle": "Aquila", "Altar": "Ara",
    "Ram": "Aries", "Charioteer": "Auriga", "Herdsman": "Bootes",
    "Chisel": "Caelum", "Giraffe": "Camelopardalis", "Crab": "Cancer",
    "Hunting Dogs": "Canes Venatici", "Greater Dog": "Canis Major",
    "Lesser Dog": "Canis Minor", "Sea Goat": "Capricornus",
    "Capricornus": "Capricornus", "Keel": "Carina", "Cassiopeia": "Cassiopeia",
    "Centaur": "Centaurus", "Cepheus": "Cepheus", "Sea Monster": "Cetus",
    "Chameleon": "Chamaeleon", "Compass": "Circinus", "Dove": "Columba",
    "Berenice's Hair": "Coma Berenices", "Southern Crown": "Corona Australis",
    "Northern Crown": "Corona Borealis", "Crow": "Corvus", "Cup": "Crater",
    "Southern Cross": "Crux", "Swan": "Cygnus", "Dolphin": "Delphinus",
    "Swordfish": "Dorado", "Dragon": "Draco", "Little Horse": "Equuleus",
    "Eridanus": "Eridanus", "Furnace": "Fornax", "Twins": "Gemini",
    "Crane": "Grus", "Hercules": "Hercules", "Pendulum Clock": "Horologium",
    "Water Snake": "Hydra", "Female Water Snake": "Hydrus",
    "Indian": "Indus", "Lizard": "Lacerta", "Lion": "Leo",
    "Lesser Lion": "Leo Minor", "Hare": "Lepus", "Scales": "Libra",
    "Wolf": "Lupus", "Lynx": "Lynx", "Lyre": "Lyra", "Table Mountain": "Mensa",
    "Microscope": "Microscopium", "Unicorn": "Monoceros", "Fly": "Musca",
    "Set Square": "Norma", "Octant": "Octans", "Serpent Bearer": "Ophiuchus",
    "Hunter": "Orion", "Peacock": "Pavo", "Winged Horse": "Pegasus",
    "Hero": "Perseus", "Phoenix": "Phoenix", "Easel": "Pictor",
    "Fish": "Pisces", "Southern Fish": "Piscis Austrinus", "Stern": "Puppis",
    "Compass Box": "Pyxis", "Net": "Reticulum", "Arrow": "Sagitta",
    "Archer": "Sagittarius", "Scorpion": "Scorpius", "Sculptor": "Sculptor",
    "Shield": "Scutum", "Serpent": "Serpens", "Sextant": "Sextans",
    "Bull": "Taurus", "Telescope": "Telescopium", "Triangle": "Triangulum",
    "Southern Triangle": "Triangulum Australe", "Toucan": "Tucana",
    "Great Bear": "Ursa Major", "Little Bear": "Ursa Minor", "Sails": "Vela",
    "Maiden": "Virgo", "Flying Fish": "Volans", "Fox": "Vulpecula",
}


def latin_name(english: str, native: str, cid: str) -> str:
    """Latin name of a figure, preferring the culture's own if it gives one."""
    nat = str(native).strip()
    if nat and nat.lower() != "nan":
        return nat
    eng = str(english).strip()
    if eng in LATIN_NAMES:
        return LATIN_NAMES[eng]
    return eng if eng and eng.lower() != "nan" else cid


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


def separations(ra_deg: np.ndarray, dec_deg: np.ndarray, ia, ib) -> np.ndarray:
    """Angular separation of each pair, per epoch. Shape (n_epoch, n_pair)."""
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    sa, ca = np.sin(dec[:, ia]), np.cos(dec[:, ia])
    sb, cb = np.sin(dec[:, ib]), np.cos(dec[:, ib])
    cosd = sa * sb + ca * cb * np.cos(ra[:, ia] - ra[:, ib])
    return np.degrees(np.arccos(np.clip(cosd, -1.0, 1.0)))


def shape_series(sep: np.ndarray, i_ref: int):
    """Scale, total change and shape distortion per epoch, from separations.

    Three numbers, because two stars would otherwise fall through the cracks.

        scale       median of the ratios: the figure as a whole growing or
                    shrinking

        total       RMS relative change of the separations, scale included.
                    This is the whole alteration, and for a pair it is the only
                    thing there is: two stars three degrees apart that end up
                    fifteen degrees apart are not the same figure enlarged, they
                    are a different sky.

        distortion  RMS spread of the ratios about their median: the part that
                    no rescaling can undo. Identically zero for a pair, since a
                    single separation is its own median, which is why the shape
                    measure alone cannot be the criterion for keeping a figure.
    """
    ref = sep[i_ref]
    good = ref > 1e-9
    ratio = sep[:, good] / ref[good]
    scale = np.median(ratio, axis=1)
    total = 100.0 * np.sqrt(np.mean((ratio - 1.0) ** 2, axis=1))
    resid = ratio / scale[:, None] - 1.0
    return scale, total, 100.0 * np.sqrt(np.mean(resid ** 2, axis=1))


def crossing_epoch(epochs: np.ndarray, dist: np.ndarray, level: float,
                   i_ref: int):
    """Epoch, going back from the reference, where distortion first exceeds level."""
    order = np.argsort(epochs)[::-1]          # from the reference backwards
    for k in order:
        if epochs[k] <= epochs[i_ref] and dist[k] > level:
            return float(epochs[k])
    return np.nan


def worst_star(ra, dec, idx, i_ref, i_far):
    """The member whose removal most reduces the distortion at the far epoch."""
    if idx.size < 4:
        return None, np.nan
    base_pairs = np.triu_indices(idx.size, k=1)
    base = shape_series(separations(ra, dec, idx[base_pairs[0]],
                                    idx[base_pairs[1]]), i_ref)[2][i_far]
    best, best_drop = None, -np.inf
    for s in range(idx.size):
        keep = np.delete(np.arange(idx.size), s)
        sub = idx[keep]
        ia, ib = np.triu_indices(sub.size, k=1)
        d = shape_series(separations(ra, dec, sub[ia], sub[ib]), i_ref)[2][i_far]
        if base - d > best_drop:
            best_drop, best = base - d, s
    return best, best_drop


def project(ra_deg: np.ndarray, dec_deg: np.ndarray):
    """Orthographic projection onto the plane tangent at the figure's centroid.

    Orthographic rather than gnomonic because a gnomonic projection diverges as
    a member approaches ninety degrees from the centre, and figures like
    Eridanus are long enough for that to matter. It compresses the outskirts,
    but it does so identically in every panel, so the comparison between epochs
    stays honest.
    """
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    v = np.column_stack((np.cos(dec) * np.cos(ra),
                         np.cos(dec) * np.sin(ra),
                         np.sin(dec)))
    c = v.mean(axis=0)
    c /= np.linalg.norm(c)
    pole = np.array([0.0, 0.0, 1.0])
    east = np.cross(pole, c)
    if np.linalg.norm(east) < 1e-8:            # centroid at a celestial pole
        east = np.cross(np.array([1.0, 0.0, 0.0]), c)
    east /= np.linalg.norm(east)
    north = np.cross(c, east)
    return np.column_stack((np.degrees(v @ east), np.degrees(v @ north)))


def procrustes(P: np.ndarray, Q: np.ndarray, fit_scale: bool = False) -> np.ndarray:
    """Bring P onto Q by translation and rotation, and by scale if asked.

    Rotation goes because precession spins the whole frame and would make every
    panel turn, hiding the only thing worth looking at. Reflection stays,
    excluded, a mirrored figure being a different figure.

    Scale does not go, by default. Removing it would leave a pair of stars
    looking identical in every panel however far apart they had drifted, and
    growth is a real alteration of the pattern rather than an artefact to be
    normalised away. The panels share axes, so it shows.
    """
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    U, S, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, d]) @ Vt
    if not fit_scale:
        return Pc @ R
    denom = float((Pc ** 2).sum())
    s = (S[0] + d * S[1]) / denom if denom > 0 else 1.0
    return s * (Pc @ R)


def star_size(vmag: np.ndarray) -> np.ndarray:
    return np.clip(220.0 * 10 ** (-0.4 * (vmag - 1.0)), 6.0, 340.0)


def draw_figure_pdf(name, epochs, epochs_sel, ra, dec, mag, idx, seg_pairs,
                    i_ref, per_page, path: Path) -> None:
    """One page of panels per group of epochs, oldest first.

    Every panel is drawn on the same axes and aligned to the reference epoch, so
    what changes between panels is shape and nothing else.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    ref_xy = project(ra[i_ref, idx], dec[i_ref, idx])
    ref_xy = ref_xy - ref_xy.mean(axis=0)

    frames = []
    for i in epochs_sel:
        xy = procrustes(project(ra[i, idx], dec[i, idx]), ref_xy)
        frames.append(xy)
    lim = 1.15 * max(np.abs(np.concatenate(frames)).max(), 1e-3)

    n_pages = int(np.ceil(len(epochs_sel) / per_page))
    ncol = 2
    nrow = int(np.ceil(per_page / ncol))

    with PdfPages(path) as pdf:
        for pg in range(n_pages):
            fig, axes = plt.subplots(nrow, ncol, figsize=(9.5, 9.5 * nrow / ncol))
            axes = np.atleast_1d(axes).ravel()
            for k in range(per_page):
                j = pg * per_page + k
                ax = axes[k]
                if j >= len(epochs_sel):
                    ax.axis("off")
                    continue
                i = epochs_sel[j]
                xy = frames[j]
                for a, b in seg_pairs:
                    ax.plot(xy[[a, b], 0], xy[[a, b], 1],
                            color="0.55", lw=1.0, zorder=1)
                ax.scatter(xy[:, 0], xy[:, 1], s=star_size(mag[i, idx]),
                           c="k", zorder=2)
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{epochs[i]:+.0f} kyr", fontsize=11)
            fig.suptitle(f"{name} — pagina {pg + 1} di {n_pages}, "
                         f"dalla piu' antica; scala e rotazione rimosse",
                         fontsize=12)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Shape deformation of constellations under proper motion.")
    p.add_argument("--trajectories", required=True,
                   help="trajectories.csv or fullcat.csv.gz")
    p.add_argument("--skycultures", required=True,
                   help="directory of CSV tables from stellarium_skycultures.py")
    p.add_argument("--culture", default="modern_iau",
                   help="sky culture supplying the figures; the physics does "
                        "not depend on it, which is checkable by rerunning with "
                        "another one")
    p.add_argument("--outdir", default="drift")
    p.add_argument("--ref-kyr", type=float, default=0.0,
                   help="reference epoch the shapes are compared against")
    p.add_argument("--min-stars", type=int, default=2,
                   help="skip figures with fewer catalogued stars. Two is the "
                        "floor: a pair has one separation, so no shape "
                        "distortion, but its total change is real and is the "
                        "whole of what happened to it. Shape needs three, the "
                        "leave-one-out four; both are skipped below that "
                        "instead of dropping the figure.")
    p.add_argument("--panel-step", type=float, default=10.0,
                   help="spacing between drawn panels, kyr")
    p.add_argument("--panels-per-page", type=int, default=4)
    p.add_argument("--pages", type=int, default=5)
    p.add_argument("--no-figures", action="store_true",
                   help="skip the per-constellation drawings")
    args = p.parse_args()

    outdir = Path(args.outdir)
    (outdir / "curves").mkdir(parents=True, exist_ok=True)
    (outdir / "figure").mkdir(parents=True, exist_ok=True)

    log(f"Reading {args.trajectories} ...")
    # Only the columns this program touches, and through the cache: the file
    # carries two dozen and parsing the rest costs time and memory for nothing.
    wanted = ["HIP", "ra_deg", "dec_deg", "distance_pc", "Vmag", "NAME", "Bayer"]
    probe = pd.read_csv(args.trajectories, nrows=0).columns   # header only
    epoch_col = ("epoch_kyr_from_year0" if "epoch_kyr_from_year0" in probe
                 else "epoch_kyr")
    traj = read_star_table(args.trajectories, [epoch_col] + wanted)
    traj = normalise_epoch(traj)
    traj = traj.drop_duplicates(["HIP", "epoch"])
    if "lat_deg" in probe:
        log("  NOTA: questo file ha una riga per latitudine; le colonne usate "
            "qui non dipendono dalla latitudine, quindi trajectories.csv "
            "contiene lo stesso e con 27 volte meno righe.")

    ra_w = traj.pivot(index="epoch", columns="HIP", values="ra_deg").sort_index()
    dec_w = traj.pivot(index="epoch", columns="HIP", values="dec_deg").sort_index()
    dist_w = traj.pivot(index="epoch", columns="HIP", values="distance_pc").sort_index()
    mag_w = traj.pivot(index="epoch", columns="HIP", values="Vmag").sort_index()
    mag = mag_w.to_numpy(dtype=float)
    epochs = ra_w.index.to_numpy(dtype=float)
    all_hips = ra_w.columns.to_numpy()
    ra = ra_w.to_numpy(dtype=float)
    dec = dec_w.to_numpy(dtype=float)
    dpc = dist_w.to_numpy(dtype=float)
    log(f"  {all_hips.size} stars, {epochs.size} epochs "
        f"[{epochs.min():+.1f}, {epochs.max():+.1f}] kyr")

    i_ref = int(np.argmin(np.abs(epochs - args.ref_kyr)))
    i_far = int(np.argmin(epochs))
    log(f"Reference epoch {epochs[i_ref]:+.1f} kyr | "
        f"farthest {epochs[i_far]:+.1f} kyr")

    labels = {}
    lab = traj.groupby("HIP")[["NAME", "Bayer"]].first()
    for hip, r in lab.iterrows():
        name, bayer = str(r["NAME"]).strip(), str(r["Bayer"]).strip()
        labels[hip] = (name if name and name != "nan"
                       else bayer if bayer and bayer != "nan"
                       else f"HIP {int(hip)}")

    members = pd.read_csv(Path(args.skycultures) / "members.csv")
    cons = pd.read_csv(Path(args.skycultures) / "constellations.csv")
    segs = pd.read_csv(Path(args.skycultures) / "segments.csv")
    sub = members[members["culture"] == args.culture]
    if sub.empty:
        avail = ", ".join(sorted(members["culture"].unique())[:12])
        raise SystemExit(f"Culture '{args.culture}' not found. Available: {avail} ...")
    cc = cons[cons["culture"] == args.culture]
    names = {r["constellation_id"]: latin_name(r.get("name_english", ""),
                                               r.get("name_native", ""),
                                               r["constellation_id"])
             for _, r in cc.iterrows()}
    segs = segs[segs["culture"] == args.culture]

    # Panels run back from the reference at a fixed spacing, oldest drawn first.
    n_panels = args.panels_per_page * args.pages
    want = epochs[i_ref] - args.panel_step * np.arange(n_panels)
    panel_idx = sorted({int(np.argmin(np.abs(epochs - t)))
                        for t in want if t >= epochs.min() - 1e-9})
    log(f"Pannelli: {len(panel_idx)} da {epochs[panel_idx[0]]:+.0f} a "
        f"{epochs[panel_idx[-1]]:+.0f} kyr, {args.panels_per_page} per pagina")

    log(f"Culture: {args.culture} | "
        f"{sub['constellation_id'].nunique()} figures")
    log("")

    rows = []
    skipped = []
    for cid, grp in sub.groupby("constellation_id"):
        hips = np.sort(grp["HIP"].unique())
        idx = np.flatnonzero(np.isin(all_hips, hips))
        if idx.size < args.min_stars:
            # Recorded rather than dropped in silence: which figures fall out,
            # and with how many stars, is what says whether the cut is the
            # threshold's fault or the catalogue's.
            skipped.append({
                "culture": args.culture,
                "constellation_id": cid,
                "name": names.get(cid, cid),
                "n_stars_catalogue": int(idx.size),
                "n_stars_figure": int(len(hips)),
            })
            continue

        ia, ib = np.triu_indices(idx.size, k=1)
        sep = separations(ra, dec, idx[ia], idx[ib])
        scale, total, dist = shape_series(sep, i_ref)

        pd.DataFrame({
            "epoch_kyr": epochs,
            "scale": scale,
            "total_change_pct": total,
            "distortion_pct": dist,
            "mean_sep_deg": sep.mean(axis=1),
        }).to_csv(outdir / "curves" / f"{cid.replace(' ', '_')}.csv",
                  index=False, float_format="%.5f")

        s, drop = worst_star(ra, dec, idx, i_ref, i_far)
        driver = labels.get(all_hips[idx[s]]) if s is not None else ""
        driver_pc = float(dpc[i_ref, idx[s]]) if s is not None else np.nan

        nome = names.get(cid, cid)
        if not args.no_figures:
            pos = {int(h): k for k, h in enumerate(all_hips[idx])}
            sp = segs[segs["constellation_id"] == cid]
            pairs = [(pos[int(a)], pos[int(b)])
                     for a, b in zip(sp["HIP_a"], sp["HIP_b"])
                     if int(a) in pos and int(b) in pos]
            safe = "".join(ch if ch.isalnum() or ch in "-_" else "_"
                           for ch in nome)
            draw_figure_pdf(nome, epochs, panel_idx, ra, dec, mag, idx, pairs,
                            i_ref, args.panels_per_page,
                            outdir / "figure" / f"{safe}.pdf")

        row = {
            "culture": args.culture,
            "constellation_id": cid,
            "name": nome,
            "n_stars": idx.size,
            "n_stars_figure": len(hips),
            "mean_sep_ref_deg": float(sep[i_ref].mean()),
            "nearest_pc": float(np.nanmin(dpc[i_ref, idx])),
            "median_pc": float(np.nanmedian(dpc[i_ref, idx])),
            "total_change_far_pct": float(total[i_far]),
            "distortion_far_pct": float(dist[i_far]),
            "scale_far": float(scale[i_far]),
            "driver_star": driver,
            "driver_distance_pc": driver_pc,
            "driver_drop_pct": float(drop) if s is not None else np.nan,
        }
        for lv in CROSS_LEVELS:
            row[f"epoch_cross_{lv:.0f}pct"] = crossing_epoch(epochs, dist, lv, i_ref)
        rows.append(row)

    if not rows:
        raise SystemExit("No figure had enough catalogued stars.")

    out = pd.DataFrame(rows).sort_values("total_change_far_pct")
    out.to_csv(outdir / "summary.csv", index=False)

    if skipped:
        sk = pd.DataFrame(skipped).sort_values(
            ["n_stars_catalogue", "name"], ascending=[False, True])
        sk.to_csv(outdir / "skipped.csv", index=False)
        log("")
        log(f"  Figure escluse: {len(sk)} su "
            f"{len(sk) + len(out)}, per stelle insufficienti nel catalogo")
        for n in sorted(sk["n_stars_catalogue"].unique(), reverse=True):
            names_n = ", ".join(sk.loc[sk["n_stars_catalogue"] == n, "name"])
            log(f"    con {n} stelle: {names_n}")
        log(f"    Il limite e' la profondita' del catalogo, non la soglia: una")
        log(f"    figura senza almeno tre stelle non ha una forma da misurare.")

    far = epochs[i_far]
    log(f"  Alterazione a {far:+.0f} kyr, dalle piu' stabili alle piu' alterate.")
    log(f"  totale = cambiamento complessivo, scala inclusa; forma = cio' che")
    log(f"  nessun riscalamento annulla, ed e' zero per una coppia.")
    log(f"    {'costellazione':22s} {'n':>3s} {'totale':>8s} {'forma':>8s} "
        f"{'scala':>6s} {'vicina':>7s} {'responsabile':>16s}")
    for _, r in out.iterrows():
        drv = str(r['driver_star'])[:16] if str(r['driver_star']) else "-"
        log(f"    {str(r['name'])[:22]:22s} {r['n_stars']:3d} "
            f"{r['total_change_far_pct']:7.1f}% {r['distortion_far_pct']:7.1f}% "
            f"{r['scale_far']:6.2f} {r['nearest_pc']:6.0f} {drv:>16s}")

    # The mechanism, stated as a number rather than an assertion: if the nearest
    # member governs the deformation, the two must track each other.
    ok = out["nearest_pc"].notna() & out["distortion_far_pct"].notna()
    if ok.sum() > 3:
        rho = np.corrcoef(np.log10(out.loc[ok, "nearest_pc"]),
                          np.log10(out.loc[ok, "distortion_far_pct"] + 1e-3))[0, 1]
        log("")
        log(f"  Correlazione log-log fra distanza della stella piu' vicina e "
            f"deformazione: {rho:+.2f}")
        log(f"    (negativa e forte = la vicinanza di un membro governa la "
            f"deformazione)")

    # A ranked bar chart rather than forty overlaid curves: the question the
    # summary answers is which figures survive and which do not, and one line
    # per constellation made that unreadable.
    fig, ax = plt.subplots(figsize=(9, max(6, 0.26 * len(out))))
    y = np.arange(len(out))
    col = plt.cm.viridis(np.clip(np.log10(out["nearest_pc"]) / 2.6, 0, 1))
    ax.barh(y, out["total_change_far_pct"], color=col)
    ax.set_yticks(y)
    ax.set_yticklabels(out["name"], fontsize=8)
    ax.invert_yaxis()
    for lv in CROSS_LEVELS:
        ax.axvline(lv, color="k", lw=0.7, ls=":")
        ax.text(lv, -0.8, f"{lv:.0f}%", fontsize=7, ha="center")
    ax.set_xscale("log")
    ax.set_xlabel(f"Alterazione totale a {epochs[i_far]:+.0f} kyr [%]  "
                  f"(scala logaritmica)")
    ax.set_title(f"{args.culture} — quali figure reggono il moto proprio\n"
                 f"colore = distanza del membro piu' vicino "
                 f"(scuro = vicino, chiaro = lontano)", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "classifica.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)

    log("")
    log(f"Written {outdir}/summary.csv ({len(out)} figure), "
        f"{outdir}/classifica.pdf")
    if not args.no_figures:
        log(f"        {outdir}/figure/*.pdf  ({len(out)} costellazioni, "
            f"{args.pages} pagine ciascuna)")


if __name__ == "__main__":
    main()
