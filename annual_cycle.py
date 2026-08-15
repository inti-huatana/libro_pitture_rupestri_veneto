#!/usr/bin/env python3
"""
annual_cycle.py

What a star does in the course of a year, at one latitude, through time.

Three questions that look separate come from the same rows of the same file, and
are three readings of one thing.

    imperishable    which stars never set. The Egyptians called them ihmw-sk,
                    those that know no destruction, and the category recurs in
                    cultures with no contact between them. Membership is not
                    fixed: precession carries stars into that circle and out of
                    it, so a tradition naming a star as one that never sets is
                    datable in the same way a canon is. If it set at that
                    latitude at that epoch, the tradition is older or younger.

    disappearance   how long a star spends too near the Sun to be seen. The gap
                    between its heliacal setting and its heliacal rising is the
                    interval during which it is gone, and for Sirius in Egypt it
                    ran to some seventy days -- the dying and returning that
                    orders the whole calendar there. Some stars vanish for a
                    season, others for barely a fortnight.

    calendar drift  how fast a star ceases to mark the date it used to mark.
                    Precession moves a heliacal rising by about one day every
                    seventy-two years, so a marker holds for a lifetime without
                    visible error, is off by days after ten generations, and by a
                    fortnight after a millennium. This is the quantitative answer
                    to whether a stellar calendar could be kept at all, and for
                    how long between recalibrations.

Reads the per-latitude catalogue, which carries the visibility class and the
four event days, not the trajectory file.

Usage
-----
    python3 annual_cycle.py --fullcat BSGRID/fullcat.csv --lat 45 --outdir annual
"""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd

from star_table import (normalise_epoch, pivot_epoch_star,
                        read_star_table, star_labels)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TROPICAL_YEAR_D = 365.2422

# Human spans the drift is quoted over. A generation is taken at 25 years, the
# figure conventionally used for oral transmission.
DRIFT_SPANS_YEARS = (25, 100, 250, 1000)


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


def gap_days(rise_day, set_day):
    """Days from heliacal setting to the next heliacal rising.

    The star sets in the evening twilight, is lost in the Sun's glare, and comes
    back before dawn some weeks later. The interval wraps the year end, so it is
    taken modulo the tropical year rather than by subtraction.
    """
    d = (np.asarray(rise_day, dtype=float) - np.asarray(set_day, dtype=float))
    return np.mod(d, TROPICAL_YEAR_D)


def contiguous_spells(epochs, mask, min_span):
    """Stretches of epochs over which a condition holds without a break."""
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
        description="Circumpolarity, invisibility and calendar drift of stars.")
    p.add_argument("--fullcat", required=True,
                   help="fullcat.csv from bright_star_grid.py, with the "
                        "visibility class and the heliacal event days")
    p.add_argument("--lat", type=float, default=45.0,
                   help="latitude to analyse; must be present in the file")
    p.add_argument("--outdir", default="annual")
    p.add_argument("--vmax", type=float, default=3.0,
                   help="faintest star considered")
    p.add_argument("--min-span", type=float, default=1.0,
                   help="shortest spell of circumpolarity reported, kyr")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log(f"Leggo {args.fullcat} ...")
    # Only the columns this program uses, and the latitude one only if the
    # file carries it: fullcat holds a row per latitude and is the largest file
    # in the set, so its width costs real time.
    have = pd.read_csv(args.fullcat, nrows=0).columns
    cols = ["epoch_kyr_from_year0", "HIP", "Vmag", "visibility",
            "heliacal_rising_day", "heliacal_setting_day", "NAME", "Bayer"]
    cols = [c for c in cols if c in have]
    if "lat_deg" in have:
        cols.append("lat_deg")
    df = read_star_table(args.fullcat, cols, verbose=True)
    df = normalise_epoch(df)
    if "lat_deg" in df.columns:
        lats = np.sort(df["lat_deg"].unique())
        pick = float(lats[np.argmin(np.abs(lats - args.lat))])
        if abs(pick - args.lat) > 1e-6:
            log(f"  latitudine {args.lat}° non nel file, uso {pick}° "
                f"(disponibili: {', '.join(f'{x:g}' for x in lats)})")
        df = df[df["lat_deg"] == pick]
        lat_used = pick
    else:
        lat_used = args.lat
        log(f"  il file non ha colonna lat_deg: assumo {lat_used}°")

    df = df.drop_duplicates(["HIP", "epoch"])
    df = df[df["Vmag"] <= args.vmax]

    labels = star_labels(df)
    epochs, hips, arr = pivot_epoch_star(
        df, ["visibility", "heliacal_rising_day", "heliacal_setting_day"])
    V = arr["visibility"]
    R = arr["heliacal_rising_day"]
    S = arr["heliacal_setting_day"]
    log(f"  {hips.size} stelle V<{args.vmax}, {epochs.size} epoche "
        f"[{epochs.min():+.1f}, {epochs.max():+.1f}] kyr, lat {lat_used:+.1f}°")

    circ = (V == "CIRCUMPOLARE")
    seen = (V == "VISIBILE")

    # ---- 1. the imperishable ---------------------------------------------
    rows = []
    for k in range(hips.size):
        for lo, hi, seg in contiguous_spells(epochs, circ[:, k], args.min_span):
            rows.append({"HIP": int(hips[k]), "star": labels.get(int(hips[k]), ""),
                         "epoch_start_kyr": float(lo), "epoch_end_kyr": float(hi),
                         "duration_kyr": float(hi - lo)})
    imp = pd.DataFrame(rows)
    if not imp.empty:
        imp = imp.sort_values(["epoch_start_kyr", "star"])
        imp.to_csv(outdir / "imperishable_spells.csv", index=False,
                   float_format="%.3f")

    n_circ = circ.sum(axis=1)
    now = int(np.argmax(epochs))
    always = [labels.get(int(hips[k]), "") for k in range(hips.size)
              if circ[:, k].all()]
    never = [labels.get(int(hips[k]), "") for k in range(hips.size)
             if not circ[:, k].any()]
    changed = hips.size - len(always) - len(never)
    log("")
    log(f"  Stelle che non tramontano mai a {lat_used:+.1f}°: "
        f"{int(n_circ[now])} oggi, da {n_circ.min()} a {n_circ.max()} nel periodo")
    log(f"    sempre imperiture: {len(always)} | mai: {len(never)} | "
        f"entrate o uscite dalla categoria: {changed}")
    if changed:
        log("    quelle che cambiano stato sono le databili: se una tradizione")
        log("    ne dice una che non tramonta mai, l'epoca e' vincolata.")

    # ---- 2. the disappearance --------------------------------------------
    gap = np.where(seen, gap_days(R, S), np.nan)
    gp = pd.DataFrame({
        "epoch_kyr": np.repeat(epochs, hips.size),
        "HIP": np.tile(hips, epochs.size),
        "gap_days": gap.ravel(),
    })
    gp["star"] = gp["HIP"].map(lambda h: labels.get(int(h), ""))
    gp.dropna(subset=["gap_days"]).to_csv(outdir / "invisibility.csv",
                                          index=False, float_format="%.3f")

    med = np.nanmedian(gap, axis=0)
    ok = np.isfinite(med)
    log("")
    log("  Durata mediana della sparizione annuale (giorni), per stella:")
    srt = np.argsort(med[ok])[::-1]
    kk = np.flatnonzero(ok)
    for k in kk[srt][:8]:
        log(f"    {labels.get(int(hips[k]), ''):16s} {med[k]:5.1f} g   piu' lunga")
    for k in kk[srt][-5:]:
        log(f"    {labels.get(int(hips[k]), ''):16s} {med[k]:5.1f} g   piu' breve")

    # ---- 3. calendar drift ------------------------------------------------
    # Rate of change of the heliacal rising date, days per century, measured
    # from the series rather than assumed from the precession rate.
    drift = np.full(hips.size, np.nan)
    for k in range(hips.size):
        m = np.isfinite(R[:, k])
        if m.sum() < 3:
            continue
        e, y = epochs[m], np.unwrap(R[m, k], period=TROPICAL_YEAR_D)
        drift[k] = np.polyfit(e, y, 1)[0] / 10.0     # days per century
    dr = pd.DataFrame({
        "HIP": hips,
        "star": [labels.get(int(h), "") for h in hips],
        "drift_days_per_century": drift,
    })
    for span in DRIFT_SPANS_YEARS:
        dr[f"drift_{span}yr_days"] = np.abs(drift) * span / 100.0
    dr.dropna(subset=["drift_days_per_century"]).to_csv(
        outdir / "calendar_drift.csv", index=False, float_format="%.4f")

    md = np.nanmedian(np.abs(drift))
    log("")
    log(f"  Deriva del calendario stellare a {lat_used:+.1f}°: mediana "
        f"{md:.2f} giorni per secolo")
    for span in DRIFT_SPANS_YEARS:
        log(f"    dopo {span:5d} anni: {md * span / 100.0:6.2f} giorni di scarto")
    log("    un marcatore stagionale regge una vita senza errore visibile,")
    log("    sbaglia di giorni dopo dieci generazioni, di due settimane in un")
    log("    millennio: ecco l'intervallo fra due ricalibrazioni.")

    # ---- figures -----------------------------------------------------------
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(11, 11))

    a1.plot(epochs, n_circ, lw=1.4, color="navy")
    a1.set_ylabel("stelle circumpolari")
    a1.set_xlabel("Epoca [kyr dall'anno 0]")
    a1.set_title(f"Le imperiture a {lat_used:+.1f}°: quante non tramontano mai\n"
                 f"({len(always)} sempre, {len(never)} mai, {changed} entrano "
                 f"o escono)")
    a1.grid(alpha=0.3)

    fin = np.isfinite(med)
    a2.hist(med[fin], bins=25, color="darkorange", edgecolor="k", linewidth=0.4)
    a2.set_xlabel("durata mediana della sparizione annuale [giorni]")
    a2.set_ylabel("stelle")
    a2.set_title("Quanto a lungo ciascuna stella resta invisibile nel Sole")
    a2.grid(alpha=0.3)

    for span, colour in zip(DRIFT_SPANS_YEARS,
                            ("0.7", "0.5", "darkorange", "crimson")):
        a3.axhline(md * span / 100.0, color=colour, lw=1.2, ls="--",
                   label=f"{span} anni: {md * span / 100.0:.1f} g")
    fd = np.abs(drift[np.isfinite(drift)])
    a3.hist(fd, bins=25, color="teal", edgecolor="k", linewidth=0.4,
            orientation="vertical")
    a3.set_xlabel("deriva della levata eliaca [giorni per secolo]")
    a3.set_ylabel("stelle")
    a3.set_title(f"Deriva del calendario: mediana {md:.2f} g/secolo")
    a3.legend(fontsize=8, title="scarto accumulato")
    a3.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "annual_cycle.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    # Who is imperishable, and when.
    ever = np.flatnonzero(circ.any(axis=0) & ~circ.all(axis=0))
    if ever.size:
        fig, ax = plt.subplots(figsize=(12, max(4, 0.3 * ever.size)))
        for y, k in enumerate(ever):
            m = circ[:, k]
            ax.scatter(epochs[m], np.full(m.sum(), y), s=5, c="navy")
        ax.set_yticks(range(ever.size))
        ax.set_yticklabels([labels.get(int(hips[k]), "") for k in ever],
                           fontsize=7)
        ax.set_xlabel("Epoca [kyr dall'anno 0]")
        ax.set_title(f"Stelle che entrano ed escono dalla categoria delle "
                     f"imperiture a {lat_used:+.1f}°\n"
                     f"(quelle sempre o mai circumpolari sono omesse)")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / "imperishable.png", dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

    log("")
    log(f"Scritti {outdir}/imperishable_spells.csv, invisibility.csv, "
        f"calendar_drift.csv")
    log(f"        {outdir}/annual_cycle.png, imperishable.png")


if __name__ == "__main__":
    main()
