#!/usr/bin/env python3
"""
asterisms.py

How long a figure in the sky stays the figure it was.

constellation_drift.py measures deformation over the 88 IAU constellations,
which are administrative areas rather than things anyone ever pointed at. What
people actually inherit are asterisms: the Dipper, the Belt, the Summer
Triangle, the Teapot, the Cross. This program takes those, named and with their
members listed by HIP, and asks the question the other one could not: given
that they come apart, after how long does the coming apart become visible, and
is the answer any different from what a random group of stars would give.

The measurement. At each epoch the members are projected gnomonically about
their own centroid and matched to their present configuration by the best
rotation and scale -- reflection excluded, since a mirrored figure is not the
same figure. What is left over is the displacement of each member from where
the inherited picture puts it, and the largest of those is the number that
matters: one star visibly out of place is enough to break a story about the
shape, while a uniform expansion of the whole pattern is not, which is exactly
why the scale is fitted out rather than measured.

The threshold. A figure is called broken when that largest residual exceeds the
greater of half a degree -- a displacement no one could miss, being the width of
the Moon -- and a tenth of the figure's own size, which is what makes the test
mean the same thing for the Belt, three degrees long, and the Dipper, which is
twenty-five. Reporting which of the two bounds is binding is part of the output,
because it says whether the answer is set by the eye or by the geometry.

Two null hypotheses, and they ask different things.

    random groups   Draw a patch of sky at random, of the same angular size,
                    and take the same number of stars of the same brightness.
                    This gives the lifetime of a chance asterism, a number
                    which as far as I can tell does not exist anywhere, and
                    which is the ceiling on how long any figurative tradition
                    can be handed down before the sky itself contradicts it.

    borrowed motion Keep the real members exactly where they are and give them
                    another star's proper motion. This isolates one thing: are
                    the durable asterisms durable because their stars travel
                    together? The prediction is sharp and checkable. Five of
                    the seven stars of the Dipper -- Merak, Phecda, Megrez,
                    Alioth, Mizar -- belong to the Ursa Major moving group and
                    share a velocity; Dubhe and Alkaid do not. If the account
                    is right, the Dipper must fail at its two ends and hold in
                    the middle, and swapping the motions must destroy that.

Usage
-----
    python3 asterisms.py --trajectories BSGRID/trajectories.csv --outdir aster
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd

from star_table import read_star_table, normalise_epoch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Asterisms as they are actually pointed at, with Hipparcos numbers. Members
# absent from the catalogue in use are reported and dropped rather than
# silently ignored: a thin catalogue removes the faint members first, and those
# are the ones that carry the shape.
ASTERISMS: dict[str, list[int]] = {
    "Grande Carro":            [54061, 53910, 58001, 59774, 62956, 65378, 67301],
    "Piccolo Carro":           [11767, 72607, 75097, 85822, 82080, 77055, 79822],
    "Cassiopea (W)":           [746, 3179, 4427, 6686, 8886],
    "Triangolo Estivo":        [91262, 102098, 97649],
    "Triangolo Invernale":     [32349, 27989, 37279],
    "Esagono Invernale":       [32349, 37279, 37826, 24608, 21421, 24436],
    "Cintura di Orione":       [25930, 26311, 26727],
    "Orione":                  [27989, 25336, 24436, 27366, 25930, 26311, 26727],
    "Pleiadi":                 [17702, 17847, 17499, 17573, 17608, 17531, 17489],
    "Iadi":                    [21421, 20205, 20455, 20889, 20894, 20885],
    "Croce del Sud":           [60718, 62434, 61084, 59747, 60260],
    "Puntatori del Centauro":  [71683, 68702],
    "Falsa Croce":             [42913, 45941, 45556, 41037],
    "Uncino dello Scorpione":  [80763, 85927, 85696, 86228, 82396, 81266,
                                80112, 78401, 78820, 78265],
    "Teiera del Sagittario":   [90185, 92855, 93506, 89931, 90496, 88635],
    "Croce del Nord":          [102098, 100453, 102488, 97165, 95947],
    "Quadrato di Pegaso":      [113963, 113881, 1067, 677],
    "Testa del Drago":         [87833, 85670, 87585, 87833],
    "Falce del Leone":         [49669, 50583, 54872, 57632],
    "Corona Boreale":          [76267, 75695, 76952, 77512, 78159, 78493],
    "Vela dell'Argo":          [42913, 44816, 45941, 39953],
    "Cefeo":                   [105199, 106032, 109492, 110991, 112724],
}

# Members of the Ursa Major moving group among the Dipper's seven. Used only to
# print the check described in the docstring, never in the computation.
UMA_MOVING_GROUP = {53910, 58001, 59774, 62956, 65378}


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def to_unit(ra_deg, dec_deg):
    a, d = np.radians(ra_deg), np.radians(dec_deg)
    return np.stack([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a),
                     np.sin(d)], axis=-1)


def gnomonic(v: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Project directions onto the plane tangent at centre, in degrees.

    Gnomonic rather than orthographic because it is the projection that carries
    great circles into straight lines, so a figure judged straight by eye stays
    straight on the page. Over the few degrees an asterism spans the difference
    is small, but the choice should still be the right one.
    """
    c = centre / np.linalg.norm(centre)
    north = np.array([0.0, 0.0, 1.0])
    e = np.cross(north, c)
    n_e = np.linalg.norm(e)
    if n_e < 1e-9:                       # centroid at a pole: any basis will do
        e = np.array([1.0, 0.0, 0.0])
        n_e = 1.0
    e = e / n_e
    n = np.cross(c, e)
    w = v @ c
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.degrees(np.stack([(v @ e) / w, (v @ n) / w], axis=-1))


def procrustes_residuals(x: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """Residual displacement of each point after the best rotation and scale.

    Reflection is excluded by forcing the determinant of the rotation positive:
    a mirrored figure is a different figure, and letting the fit flip it would
    hide exactly the deformation being looked for.
    """
    a = x - x.mean(axis=0)
    b = x0 - x0.mean(axis=0)
    u, s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1.0, d]) @ vt
    denom = float(np.sum(a * a))
    scale = float(np.sum(s * np.array([1.0, d]))) / denom if denom > 0 else 1.0
    return np.linalg.norm(a @ r * scale - b, axis=1)


def figure_size(x: np.ndarray) -> float:
    """Greatest distance between two members, degrees: the figure's own scale."""
    if x.shape[0] < 2:
        return np.nan
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    return float(d.max())


def drift_series(V: np.ndarray, i_ref: int):
    """Largest residual, and the overall scale change, epoch by epoch.

    V has shape (epochs, members, 3) and holds unit vectors. The reference
    configuration is the one at i_ref, normally the present.
    """
    n_ep, n_st, _ = V.shape
    c0 = V[i_ref].sum(axis=0)
    x0 = gnomonic(V[i_ref], c0)
    size0 = figure_size(x0)

    worst = np.zeros(n_ep)
    scale = np.ones(n_ep)
    for i in range(n_ep):
        c = V[i].sum(axis=0)
        x = gnomonic(V[i], c)
        if n_st >= 3:
            worst[i] = float(procrustes_residuals(x, x0).max())
        else:
            worst[i] = 0.0            # a pair has no shape, only a separation
        s = figure_size(x)
        scale[i] = s / size0 if size0 and np.isfinite(size0) else np.nan
    return worst, scale, size0


def lifetime(epochs: np.ndarray, worst: np.ndarray, i_ref: int,
             threshold: float) -> float:
    """Span of epochs around the reference over which the figure holds.

    Taken as the contiguous run containing the reference epoch, so that a later
    accidental return to a similar shape does not count as the figure having
    survived through the intervening deformation.
    """
    ok = worst <= threshold
    if not ok[i_ref]:
        return 0.0
    lo = i_ref
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    hi = i_ref
    while hi < ok.size - 1 and ok[hi + 1]:
        hi += 1
    return float(abs(epochs[hi] - epochs[lo]))


# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Deformation and lifetime of real and random asterisms.")
    p.add_argument("--trajectories", required=True)
    p.add_argument("--outdir", default="aster")
    p.add_argument("--ref-epoch", type=float, default=0.0,
                   help="epoch the inherited figure is taken from, kyr")
    p.add_argument("--break-deg", type=float, default=0.5,
                   help="absolute displacement that breaks a figure, deg")
    p.add_argument("--break-frac", type=float, default=0.10,
                   help="displacement as a fraction of the figure's own size")
    p.add_argument("--vmax", type=float, default=4.0,
                   help="faintest star a random group may draw")
    p.add_argument("--n-random", type=int, default=400,
                   help="random groups drawn per real asterism")
    p.add_argument("--seed", type=int, default=20260815)
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    log(f"Leggo {args.trajectories} ...")
    traj = read_star_table(args.trajectories,
                           ["epoch_kyr_from_year0", "HIP", "ra_deg", "dec_deg",
                            "Vmag", "NAME", "Bayer"])
    traj = normalise_epoch(traj).drop_duplicates(["HIP", "epoch"])

    ra_w = traj.pivot(index="epoch", columns="HIP", values="ra_deg").sort_index()
    de_w = traj.pivot(index="epoch", columns="HIP", values="dec_deg").sort_index()
    mg_w = traj.pivot(index="epoch", columns="HIP", values="Vmag").sort_index()
    epochs = ra_w.index.to_numpy(dtype=float)
    hips = ra_w.columns.to_numpy()
    index = {int(h): k for k, h in enumerate(hips)}
    UV = to_unit(ra_w.to_numpy(dtype=float), de_w.to_numpy(dtype=float))
    MG = mg_w.to_numpy(dtype=float)
    i_ref = int(np.argmin(np.abs(epochs - args.ref_epoch)))
    log(f"  {hips.size} stelle, {epochs.size} epoche "
        f"[{epochs.min():+.1f}, {epochs.max():+.1f}] kyr; "
        f"riferimento a {epochs[i_ref]:+.1f} kyr")

    lab = traj.groupby("HIP")[["NAME", "Bayer"]].first()
    labels = {}
    for hip, r in lab.iterrows():
        nm, by = str(r["NAME"]).strip(), str(r["Bayer"]).strip()
        labels[int(hip)] = (nm if nm and nm != "nan"
                            else by if by and by != "nan" else f"HIP {int(hip)}")

    # ---- membership --------------------------------------------------------
    members, mem_rows = {}, []
    for name, hl in ASTERISMS.items():
        seen, keep, missing = set(), [], []
        for h in hl:
            if h in seen:
                continue
            seen.add(h)
            (keep if h in index else missing).append(h)
        members[name] = keep
        for h in keep:
            mem_rows.append({"asterismo": name, "HIP": h,
                             "star": labels.get(h, ""),
                             "Vmag": float(MG[i_ref, index[h]]),
                             "presente": True})
        for h in missing:
            mem_rows.append({"asterismo": name, "HIP": h, "star": "",
                             "Vmag": np.nan, "presente": False})
        if missing:
            log(f"  {name}: mancano dal catalogo {missing} "
                f"({len(keep)} membri su {len(seen)})")
    pd.DataFrame(mem_rows).to_csv(outdir / "members.csv", index=False)

    usable = {k: v for k, v in members.items() if len(v) >= 2}
    dropped = [k for k, v in members.items() if len(v) < 2]
    if dropped:
        log(f"  Saltati per meno di 2 membri: {dropped}")

    # ---- real asterisms ----------------------------------------------------
    log("")
    log("Deformazione degli asterismi reali ...")
    drift_rows, life_rows, curves = [], [], {}
    for name, hl in usable.items():
        cols = [index[h] for h in hl]
        worst, scale, size0 = drift_series(UV[:, cols, :], i_ref)
        thr_abs, thr_rel = args.break_deg, args.break_frac * size0
        thr = max(thr_abs, thr_rel)
        life = lifetime(epochs, worst, i_ref, thr)
        curves[name] = (worst, thr)
        for i in range(epochs.size):
            drift_rows.append({"asterismo": name, "epoch_kyr": float(epochs[i]),
                               "max_residuo_deg": float(worst[i]),
                               "scala": float(scale[i])})
        life_rows.append({
            "asterismo": name, "n_membri": len(hl),
            "dimensione_deg": size0,
            "soglia_deg": thr,
            "soglia_vincolante": "occhio" if thr_abs >= thr_rel else "figura",
            "vita_kyr": life,
            "residuo_max_deg": float(worst.max()),
            "scala_min": float(np.nanmin(scale)),
            "scala_max": float(np.nanmax(scale)),
        })
    pd.DataFrame(drift_rows).to_csv(outdir / "asterism_drift.csv", index=False,
                                    float_format="%.5f")
    life = pd.DataFrame(life_rows).sort_values("vita_kyr", ascending=False)

    log("")
    log("  Vita delle figure, dalla piu' longeva:")
    for _, r in life.iterrows():
        span = epochs.max() - epochs.min()
        tag = "  (regge tutto l'intervallo)" if r["vita_kyr"] >= span - 1e-6 else ""
        log(f"    {r['asterismo']:26s} {int(r['n_membri'])} membri, "
            f"{r['dimensione_deg']:5.1f}° di taglia, soglia "
            f"{r['soglia_deg']:4.2f}° ({r['soglia_vincolante']:6s})  ->  "
            f"{r['vita_kyr']:7.1f} kyr{tag}")

    # ---- the Dipper check --------------------------------------------------
    if "Grande Carro" in usable:
        hl = usable["Grande Carro"]
        cols = [index[h] for h in hl]
        V = UV[:, cols, :]
        c0 = V[i_ref].sum(axis=0)
        x0 = gnomonic(V[i_ref], c0)
        res_end = np.zeros(len(hl))
        for i in range(epochs.size):
            x = gnomonic(V[i], V[i].sum(axis=0))
            res_end = np.maximum(res_end, procrustes_residuals(x, x0))
        log("")
        log("  Verifica sul Grande Carro: chi si sposta e chi no")
        for k, h in enumerate(hl):
            tag = "gruppo mobile UMa" if h in UMA_MOVING_GROUP else "ESTRANEA"
            log(f"    {labels.get(h, ''):12s} residuo massimo "
                f"{res_end[k]:6.2f}°   {tag}")
        ins = np.array([res_end[k] for k, h in enumerate(hl)
                        if h in UMA_MOVING_GROUP])
        out = np.array([res_end[k] for k, h in enumerate(hl)
                        if h not in UMA_MOVING_GROUP])
        if ins.size and out.size:
            log(f"    membri del gruppo: {ins.mean():.2f}° in media; "
                f"estranee: {out.mean():.2f}°  "
                f"(rapporto {out.mean() / max(ins.mean(), 1e-9):.1f})")

    # ---- null 1: random groups --------------------------------------------
    log("")
    log(f"Gruppi casuali: {args.n_random} per asterismo ...")
    bright = np.flatnonzero(MG[i_ref] <= args.vmax)
    UVR = UV[:, bright, :]
    ref_dirs = UVR[i_ref]
    rand_rows = []
    for name, hl in usable.items():
        k = len(hl)
        size0 = float(life.loc[life["asterismo"] == name,
                               "dimensione_deg"].iloc[0])
        thr = float(life.loc[life["asterismo"] == name, "soglia_deg"].iloc[0])
        radius = math.radians(0.5 * size0) if np.isfinite(size0) else 0.1
        got = 0
        tries = 0
        while got < args.n_random and tries < 40 * args.n_random:
            tries += 1
            c = rng.normal(size=3)
            c /= np.linalg.norm(c)
            near = np.flatnonzero(ref_dirs @ c >= math.cos(radius))
            if near.size < k:
                continue
            pick = rng.choice(near, size=k, replace=False)
            worst, _, s0 = drift_series(UVR[:, pick, :], i_ref)
            if not np.isfinite(s0) or s0 <= 0:
                continue
            rand_rows.append({"asterismo": name, "n_membri": k,
                              "dimensione_deg": s0,
                              "vita_kyr": lifetime(epochs, worst, i_ref, thr),
                              "residuo_max_deg": float(worst.max())})
            got += 1
        log(f"  {name:26s} {got} gruppi in {tries} tentativi")
    rnd = pd.DataFrame(rand_rows)
    rnd.to_csv(outdir / "random_lifetimes.csv", index=False,
               float_format="%.4f")

    # ---- null 2: borrowed motion ------------------------------------------
    # The member keeps its place and takes another star's displacement field,
    # transplanted through the tangent plane at each donor's own position. All
    # displacements here are of the order of a degree, so the transplant is
    # accurate to second order and the two nulls stay comparable.
    log("")
    log("Moti prestati ...")
    disp = np.empty((epochs.size, bright.size, 2))
    for k in range(bright.size):
        disp[:, k, :] = gnomonic(UVR[:, k, :], ref_dirs[k])
    swap_rows = []
    for name, hl in usable.items():
        cols = [index[h] for h in hl]
        c0 = UV[i_ref, cols, :].sum(axis=0)
        x0 = gnomonic(UV[i_ref, cols, :], c0)
        thr = float(life.loc[life["asterismo"] == name, "soglia_deg"].iloc[0])
        for _ in range(max(args.n_random // 4, 25)):
            donors = rng.choice(bright.size, size=len(hl), replace=False)
            x = x0[None, :, :] + disp[:, donors, :]
            worst = np.array([float(procrustes_residuals(x[i], x0).max())
                              if len(hl) >= 3 else 0.0
                              for i in range(epochs.size)])
            swap_rows.append({"asterismo": name,
                              "vita_kyr": lifetime(epochs, worst, i_ref, thr),
                              "residuo_max_deg": float(worst.max())})
    swp = pd.DataFrame(swap_rows)
    swp.to_csv(outdir / "borrowed_motion.csv", index=False,
               float_format="%.4f")

    # ---- comparison --------------------------------------------------------
    comp = []
    for name in usable:
        real = float(life.loc[life["asterismo"] == name, "vita_kyr"].iloc[0])
        a = rnd[rnd["asterismo"] == name]["vita_kyr"].to_numpy()
        b = swp[swp["asterismo"] == name]["vita_kyr"].to_numpy()
        comp.append({
            "asterismo": name, "vita_reale_kyr": real,
            "casuale_mediana_kyr": float(np.median(a)) if a.size else np.nan,
            "casuale_p90_kyr": float(np.percentile(a, 90)) if a.size else np.nan,
            "p_value_casuale": float(np.mean(a >= real)) if a.size else np.nan,
            "prestata_mediana_kyr": float(np.median(b)) if b.size else np.nan,
            "p_value_prestata": float(np.mean(b >= real)) if b.size else np.nan,
        })
    comp = pd.DataFrame(comp).sort_values("p_value_casuale")
    comp.to_csv(outdir / "asterism_lifetimes.csv", index=False,
                float_format="%.5f")
    life.to_csv(outdir / "asterism_summary.csv", index=False,
                float_format="%.5f")

    log("")
    log("  Reale contro casuale (p = frazione di gruppi casuali almeno "
        "altrettanto longevi):")
    for _, r in comp.iterrows():
        log(f"    {r['asterismo']:26s} reale {r['vita_reale_kyr']:7.1f} kyr | "
            f"casuale mediana {r['casuale_mediana_kyr']:7.1f} | "
            f"p {r['p_value_casuale']:6.3f} | "
            f"moti prestati mediana {r['prestata_mediana_kyr']:7.1f} "
            f"(p {r['p_value_prestata']:6.3f})")

    if len(rnd):
        log("")
        log(f"  Vita di un asterismo casuale, su tutti i tiri: mediana "
            f"{rnd['vita_kyr'].median():.1f} kyr, "
            f"decile superiore {np.percentile(rnd['vita_kyr'], 90):.1f} kyr, "
            f"massimo {rnd['vita_kyr'].max():.1f} kyr")

    # ---- figures -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for name, (worst, thr) in curves.items():
        ax.plot(epochs, worst, lw=1.1, label=name)
    ax.axhline(args.break_deg, color="k", lw=1.0, ls="--",
               label=f"soglia assoluta {args.break_deg:.1f}°")
    ax.axvline(epochs[i_ref], color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("epoca [kyr dall'anno 0]")
    ax.set_ylabel("spostamento massimo di un membro [°]")
    ax.set_yscale("log")
    ax.set_title("Deformazione delle figure note\n"
                 "rotazione e scala tolte: resta solo la stella che non "
                 "sta piu' dove la mette il racconto")
    ax.legend(fontsize=6, ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "drift_curves.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    order = comp.sort_values("vita_reale_kyr")["asterismo"].tolist()
    fig, ax = plt.subplots(figsize=(11, max(5, 0.42 * len(order))))
    for y, name in enumerate(order):
        a = rnd[rnd["asterismo"] == name]["vita_kyr"].to_numpy()
        if a.size:
            ax.boxplot(a, positions=[y], vert=False, widths=0.6,
                       showfliers=False,
                       medianprops=dict(color="0.3"),
                       boxprops=dict(color="0.6"),
                       whiskerprops=dict(color="0.6"),
                       capprops=dict(color="0.6"))
        r = float(comp.loc[comp["asterismo"] == name,
                           "vita_reale_kyr"].iloc[0])
        ax.plot([r], [y], "o", color="firebrick", ms=7, zorder=5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("vita della figura [kyr]")
    ax.set_title("Asterismi reali (punti rossi) contro gruppi casuali della "
                 "stessa taglia e numerosita' (scatole)\n"
                 "a destra della scatola: la figura dura piu' del caso")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "asterism_lifetimes.png", dpi=args.dpi,
                bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    if len(rnd):
        ax.hist(rnd["vita_kyr"], bins=40, color="0.7",
                label="gruppi casuali")
    if len(swp):
        ax.hist(swp["vita_kyr"], bins=40, histtype="step", color="teal",
                lw=1.4, label="membri reali, moti prestati")
    for _, r in comp.iterrows():
        ax.axvline(r["vita_reale_kyr"], color="firebrick", lw=0.7, alpha=0.6)
    ax.set_xlabel("vita della figura [kyr]")
    ax.set_ylabel("conteggio")
    ax.set_title("La vita naturale di un asterismo\n"
                 "linee rosse: gli asterismi reali")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "random_lifetime_distribution.png", dpi=args.dpi,
                bbox_inches="tight")
    plt.close(fig)

    log("")
    log(f"Scritti {outdir}/members.csv, asterism_drift.csv, "
        f"asterism_summary.csv, asterism_lifetimes.csv, random_lifetimes.csv, "
        f"borrowed_motion.csv, drift_curves.png, asterism_lifetimes.png, "
        f"random_lifetime_distribution.png")


if __name__ == "__main__":
    main()
