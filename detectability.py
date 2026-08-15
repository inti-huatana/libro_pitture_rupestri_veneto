#!/usr/bin/env python3
"""
detectability.py

How long before the sky visibly disagrees with what was handed down.

Every other program in this set produces a drift: degrees per millennium, days
per century, magnitudes per hundred thousand years. None of those numbers means
anything until it is set against what an unaided eye can actually resolve, and
that comparison is what this computes. It is the filter, not a new measurement:
it takes rates we already have and returns, for each of them, the number of
years -- and of generations -- before the change exceeds the threshold of the
instrument that was available.

The thresholds are not chosen freely. Four are enough and each has a check.

    vernier         About one arcminute, for judging whether three points are
                    in line or which of two stars sits higher. The eye is far
                    better at this than at absolute position, and the standard
                    proof is naked: Mizar and Alcor are 11.8 arcminutes apart
                    and telling them apart is an ordinary feat.

    absolute        Ten arcminutes, for placing something against the horizon
                    or against a fixed mark. Refraction near the horizon is
                    34 arcminutes and fluctuates by several from day to day,
                    which sets the floor regardless of the observer.

    solar diameter  Thirty-two arcminutes. The Sun and the Moon are the rulers
                    a horizon calendar is actually read with: a shift smaller
                    than the disc is not a shift anyone reports.

    one day         The resolution of a calendar kept by counting.

Two results come out of this that reverse the usual assumptions.

The first is that the solstice is a poor marker and the equinox an excellent
one. The Sun's declination is stationary at the solstice, so its rising point
crawls: within ten days either side of it the sunrise azimuth moves by about
one solar diameter, and the date cannot be fixed better than that. At the
equinox the declination runs at 0.4 degrees a day and the rising point moves a
full solar diameter every day. An order of magnitude separates them, in the
direction opposite to the literature's habit of treating solstices as the
natural target.

The second is that precession is invisible on the horizon and loud in the
calendar. An observer watching where the solstice Sun rises from a fixed place
sees only the obliquity, which moves that point by about one solar diameter in
two and a half millennia -- silent across any oral tradition. The same observer
noting in which season a star first reappears sees precession at a day every
seventy-two years. Same sky, same physics, four orders of magnitude between the
two instruments, and it is why stellar traditions can be dated and stone
alignments almost never can.

Writes the ledger as CSV, as LaTeX tables ready to \\input, and as figures.

Usage
-----
    python3 detectability.py --catalog hip_mag4.csv --lat 45.5 --outdir detect
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
except ImportError as exc:                                    # pragma: no cover
    raise SystemExit("Missing dependency 'pyerfa'. Install with: "
                     "pip install pyerfa") from exc

TROPICAL_YEAR_D = 365.2422
GENERATION_YEARS = 25.0            # the figure conventionally used for oral transmission

# Thresholds, arcminutes. See the module docstring for what fixes each.
THRESHOLDS_ARCMIN = {
    "vernier": 1.0,
    "assoluta": 10.0,
    "disco solare": 32.0,
}
# Magnitude difference the eye can call between two stars seen together. The
# classical magnitude scale itself is built on steps of about this size.
THRESHOLD_MAG = 0.2

# Amplitude of the principal term of nutation in obliquity, arcseconds, and its
# period in years. Constant on any timescale considered here.
NUTATION_ARCSEC = 9.2
NUTATION_PERIOD_YR = 18.61

# Inclination of the lunar orbit: the same 18.6-year cycle, seen the other way.
LUNAR_INCLINATION_DEG = 5.145

# Lunar recession, metres per year, and the present semi-major axis in metres.
LUNAR_RECESSION_M_PER_YR = 0.038
LUNAR_SEMIMAJOR_M = 3.844e8


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# rates taken from the precession model rather than quoted
# --------------------------------------------------------------------------

def precession_rates(epoch_kyr: float = 0.0, span_kyr: float = 0.1):
    """General precession and the speed of the celestial pole, arcsec per year.

    Both are differenced from the long-term model over a short span rather than
    taken as constants, so that the same numbers hold at any epoch the ledger is
    asked for. The pole moves more slowly than the equinox by a factor sin(eps),
    because it runs on a small circle of that angular radius about the fixed
    pole of the ecliptic.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = {}
        for tag, t in (("a", epoch_kyr - 0.5 * span_kyr),
                       ("b", epoch_kyr + 0.5 * span_kyr)):
            y = 1000.0 * t
            out[tag] = (np.asarray(erfa.ltpequ(y), dtype=float),
                        np.asarray(erfa.ltpecl(y), dtype=float))
    (pa, ea), (pb, eb) = out["a"], out["b"]
    years = 1000.0 * span_kyr

    pole_rate = math.degrees(math.acos(np.clip(float(np.dot(pa, pb)),
                                               -1.0, 1.0))) * 3600.0 / years
    eps_a = math.degrees(math.acos(np.clip(float(np.dot(pa, ea)), -1.0, 1.0)))
    eps_b = math.degrees(math.acos(np.clip(float(np.dot(pb, eb)), -1.0, 1.0)))
    obliquity_rate = (eps_b - eps_a) * 3600.0 / years
    eps = 0.5 * (eps_a + eps_b)

    # The equinox runs faster than the pole by 1/sin(eps): same rotation, larger
    # circle. Recovering it this way keeps the two mutually consistent.
    general = pole_rate / math.sin(math.radians(eps))
    return general, pole_rate, obliquity_rate, eps


def heliacal_drift_days_per_year(general_arcsec_per_yr: float) -> float:
    """Days the heliacal date of a star moves per year of elapsed time.

    Precession carries the equinox along the ecliptic, so the Sun meets a fixed
    star a little earlier each year: the whole circle of 360 degrees corresponds
    to one tropical year of drift.
    """
    return TROPICAL_YEAR_D * general_arcsec_per_yr / (360.0 * 3600.0)


# --------------------------------------------------------------------------
# the horizon as an instrument
# --------------------------------------------------------------------------

def sun_declination(lam_rad, eps_deg: float):
    return np.arcsin(math.sin(math.radians(eps_deg)) * np.sin(lam_rad))


def rise_azimuth_deg(dec_rad, lat_deg: float):
    """Azimuth from north of the rising point, NaN where the body never rises."""
    c = np.sin(dec_rad) / math.cos(math.radians(lat_deg))
    c = np.where(np.abs(c) <= 1.0, c, np.nan)
    return np.degrees(np.arccos(c))


def date_resolution(lat_deg: float, eps_deg: float, threshold_arcmin: float,
                    max_days: int = 120):
    """Days needed for the sunrise point to move by the threshold, day by day.

    Answers the operational question directly instead of differentiating, which
    is what avoids the singularity: at the solstice the derivative vanishes and
    only a finite difference says anything. Returns one number per day of the
    year, counted from the vernal equinox.
    """
    days = np.arange(math.ceil(TROPICAL_YEAR_D))
    lam = 2.0 * math.pi * days / TROPICAL_YEAR_D
    az = rise_azimuth_deg(sun_declination(lam, eps_deg), lat_deg)
    thr = threshold_arcmin / 60.0

    out = np.full(days.size, np.nan)
    n = days.size
    for d in range(n):
        if not np.isfinite(az[d]):
            continue
        for k in range(1, max_days + 1):
            a2 = az[(d + k) % n]
            if np.isfinite(a2) and abs(a2 - az[d]) >= thr:
                out[d] = k
                break
    return days, az, out


# --------------------------------------------------------------------------

def _fmt_years(y: float) -> str:
    if not np.isfinite(y):
        return "mai"
    if y >= 1e6:
        return f"{y / 1e6:.0f} Myr"
    if y >= 1e4:
        return f"{y / 1000:.0f} kyr"
    if y >= 100:
        return f"{y:.0f}"
    return f"{y:.1f}"


def latex_table(df: pd.DataFrame, columns, headers, caption: str,
                label: str, align: str) -> str:
    """A booktabs table ready to be \\input, with no float placement forced."""
    out = ["\\begin{table}[htbp]", "\\centering", "\\small",
           f"\\begin{{tabular}}{{{align}}}", "\\toprule",
           " & ".join(headers) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        cells = []
        for c in columns:
            v = r[c]
            cells.append(v if isinstance(v, str) else
                         ("--" if not np.isfinite(v) else f"{v:.4g}"))
        out.append(" & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}",
            f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}", ""]
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(
        description="What is detectable by eye, and after how many generations.")
    p.add_argument("--catalog", default="hip_mag4.csv",
                   help="star catalogue, for the proper-motion rows")
    p.add_argument("--pm-vmax", type=float, default=4.0,
                   help="faintest star allowed to carry the proper-motion row: "
                        "past this an eye cannot keep a star apart from its "
                        "neighbours, so it cannot serve as a reference")
    p.add_argument("--lat", type=float, default=45.5,
                   help="latitude the horizon quantities are worked out at")
    p.add_argument("--epoch", type=float, default=0.0,
                   help="epoch the rates are taken at, kyr from year 0")
    p.add_argument("--outdir", default="detect")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    general, pole_rate, obl_rate, eps = precession_rates(args.epoch)
    hel = heliacal_drift_days_per_year(general)
    log(f"Epoca {args.epoch:+.1f} kyr, latitudine {args.lat:.1f}°")
    log(f"  obliquita'            {eps:.4f}°")
    log(f"  precessione generale  {general:.3f}\"/anno "
        f"(un giro in {360 * 3600 / general / 1000:.2f} kyr)")
    log(f"  moto del polo         {pole_rate:.3f}\"/anno")
    log(f"  deriva dell'obliquita' {obl_rate * 100:.2f}\"/secolo")
    log(f"  deriva eliaca         1 giorno ogni {1 / hel:.1f} anni")

    # ---- the horizon, equinox against solstice ----------------------------
    log("")
    log("Risoluzione della data sull'orizzonte ...")
    thr_sun = THRESHOLDS_ARCMIN["disco solare"]
    days, az, res = date_resolution(args.lat, eps, thr_sun)

    i_eq = 0                                     # lambda = 0, vernal equinox
    i_ss = int(round(TROPICAL_YEAR_D * 0.25))    # summer solstice
    i_ws = int(round(TROPICAL_YEAR_D * 0.75))
    log(f"  a {args.lat:.1f}°, per spostare la levata di un disco solare "
        f"({thr_sun:.0f}'):")
    log(f"    equinozio          {res[i_eq]:5.0f} giorni")
    log(f"    solstizio d'estate {res[i_ss]:5.0f} giorni")
    log(f"    solstizio d'inverno{res[i_ws]:5.0f} giorni")
    if np.isfinite(res[i_eq]) and res[i_eq] > 0:
        log(f"    il solstizio e' {res[i_ss] / res[i_eq]:.0f} volte "
            f"meno risolvente dell'equinozio")

    # The same, latitude by latitude, as a table for the book.
    rows = []
    for phi in (0.0, 10.0, 20.0, 30.0, 40.0, args.lat, 50.0, 55.0, 60.0, 65.0):
        _, _, r = date_resolution(phi, eps, thr_sun)
        rows.append({"lat_deg": phi,
                     "risoluzione_equinozio_giorni": r[i_eq],
                     "risoluzione_solstizio_giorni": r[i_ss],
                     "rapporto": (r[i_ss] / r[i_eq]
                                  if np.isfinite(r[i_eq]) and r[i_eq] > 0
                                  else np.nan)})
    hor = pd.DataFrame(rows)
    hor.to_csv(outdir / "horizon_resolution.csv", index=False,
               float_format="%.2f")

    # ---- proper motion ----------------------------------------------------
    log("")
    pm = pd.DataFrame()
    cat = Path(args.catalog)
    if cat.exists():
        c = pd.read_csv(cat)
        # pmra in these catalogues is the great-circle rate, already carrying
        # the cos(dec) factor, so the two components combine in quadrature.
        c["pm_arcsec_per_yr"] = np.hypot(c["pmra"], c["pmde"]) / 1000.0
        pm_all = c[["HIP", "NAME", "Bayer", "Vmag", "pm_arcsec_per_yr"]].copy()
        pm_all = pm_all.sort_values("pm_arcsec_per_yr", ascending=False)
        # The catalogue now reaches past the eye's limit, and the fastest
        # movers in it are faint red dwarfs that no one could have used as a
        # reference. The ledger row is therefore built at the magnitude beyond
        # which a star cannot be told apart from its neighbours, and the deeper
        # number is reported separately rather than quietly substituted.
        pm = pm_all[pm_all["Vmag"] <= args.pm_vmax].copy()
        log(f"Moti propri da {cat.name}: {len(pm_all)} stelle in tutto, "
            f"{len(pm)} con V<{args.pm_vmax}")
        log(f"  V<{args.pm_vmax}: mediana "
            f"{pm['pm_arcsec_per_yr'].median():.3f}\"/anno, massimo "
            f"{pm['pm_arcsec_per_yr'].max():.3f}\"/anno")
        r_deep = pm_all.iloc[0]
        nm_deep = str(r_deep["NAME"]).strip()
        nm_deep = nm_deep if nm_deep and nm_deep != "nan" else str(
            r_deep["Bayer"]).strip()
        log(f"  senza limite: {nm_deep} V {r_deep['Vmag']:.1f} a "
            f"{r_deep['pm_arcsec_per_yr']:.3f}\"/anno "
            f"(troppo debole per essere un riferimento)")
        for _, r in pm.head(8).iterrows():
            nm = str(r["NAME"]).strip()
            nm = nm if nm and nm != "nan" else str(r["Bayer"]).strip()
            yv = 60.0 * THRESHOLDS_ARCMIN["vernier"] / r["pm_arcsec_per_yr"]
            log(f"    {nm:20s} V {r['Vmag']:4.1f}  "
                f"{r['pm_arcsec_per_yr']:6.3f}\"/anno  ->  1' in "
                f"{yv:6.0f} anni ({yv / GENERATION_YEARS:5.1f} generazioni)")
        pm.to_csv(outdir / "proper_motions.csv", index=False,
                  float_format="%.5f")
    else:
        log(f"Catalogo {cat} assente: righe di moto proprio saltate.")

    # ---- the ledger -------------------------------------------------------
    # Each entry: what changes, what one would have to be watching to see it,
    # how fast it changes in arcminutes per year, and against which threshold.
    fastest = (pm["pm_arcsec_per_yr"].max() / 60.0) if len(pm) else np.nan
    fastest_name = ""
    if len(pm):
        r0 = pm.iloc[0]
        fastest_name = str(r0["NAME"]).strip()
        if not fastest_name or fastest_name == "nan":
            fastest_name = str(r0["Bayer"]).strip()

    # Sensitivity of the solstice sunrise azimuth to the obliquity, taken as a
    # finite difference so it holds at any latitude without the small-angle step.
    def solstice_az(e):
        return rise_azimuth_deg(
            np.array([math.radians(e)]), args.lat)[0]

    d_az_d_eps = ((solstice_az(eps + 0.01) - solstice_az(eps - 0.01)) / 0.02
                  if np.isfinite(solstice_az(eps)) else np.nan)
    obl_az_rate = abs(d_az_d_eps) * (obl_rate / 3600.0) * 60.0   # arcmin/yr

    entries = [
        ("Precessione", "data della levata eliaca di una stella",
         hel, "giorni", 1.0, "un giorno"),
        ("Precessione", "azimut di levata di una stella all'orizzonte",
         (pole_rate / 60.0) * abs(1.0 / max(math.cos(math.radians(args.lat)),
                                            1e-6)),
         "arcmin", THRESHOLDS_ARCMIN["disco solare"], "disco solare"),
        ("Precessione", "distanza fra il polo e la stella polare del momento",
         pole_rate / 60.0, "arcmin", THRESHOLDS_ARCMIN["assoluta"], "assoluta"),
        ("Obliquita'", "altezza del Sole a mezzogiorno al solstizio",
         abs(obl_rate) / 60.0, "arcmin", THRESHOLDS_ARCMIN["assoluta"],
         "assoluta"),
        ("Obliquita'", f"azimut della levata solstiziale a {args.lat:.1f}°",
         obl_az_rate, "arcmin", THRESHOLDS_ARCMIN["disco solare"],
         "disco solare"),
        ("Moto proprio", f"posizione di {fastest_name} fra le vicine",
         fastest, "arcmin", THRESHOLDS_ARCMIN["vernier"], "vernier"),
        ("Nutazione", "posizione del polo",
         0.0, "arcmin", THRESHOLDS_ARCMIN["vernier"], "vernier"),
        ("Lunistizio", "azimut estremo della levata lunare",
         0.0, "arcmin", THRESHOLDS_ARCMIN["disco solare"], "disco solare"),
        ("Recessione lunare", "diametro apparente della Luna",
         0.0, "arcmin", THRESHOLDS_ARCMIN["disco solare"], "disco solare"),
    ]

    led = []
    for phenom, observable, rate, unit, thr, thr_name in entries:
        years = np.inf if rate <= 0 else thr / rate
        led.append({"fenomeno": phenom, "osservabile": observable,
                    "tasso_per_anno": rate, "unita": unit,
                    "soglia": thr, "soglia_nome": thr_name,
                    "anni_per_rilevare": years,
                    "generazioni": years / GENERATION_YEARS})

    # The three periodic quantities are not drifts and their rows are filled in
    # by amplitude against threshold, which is the only comparison that applies.
    amp = {
        "Nutazione": NUTATION_ARCSEC / 60.0,
        "Lunistizio": 2.0 * (eps + LUNAR_INCLINATION_DEG) * 60.0,
        "Recessione lunare": (31.1 * 2.0 * LUNAR_RECESSION_M_PER_YR
                              * 200000.0 / LUNAR_SEMIMAJOR_M),
    }
    led = pd.DataFrame(led)
    for k, a in amp.items():
        m = led["fenomeno"] == k
        led.loc[m, "tasso_per_anno"] = np.nan
        led.loc[m, "ampiezza_arcmin"] = a
        led.loc[m, "anni_per_rilevare"] = np.where(
            a >= led.loc[m, "soglia"], NUTATION_PERIOD_YR, np.inf)
        led.loc[m, "generazioni"] = led.loc[m, "anni_per_rilevare"] / GENERATION_YEARS
        led.loc[m, "rapporto_ampiezza_soglia"] = a / led.loc[m, "soglia"]
    led.loc[led["fenomeno"] == "Recessione lunare", "anni_per_rilevare"] = np.inf
    led.loc[led["fenomeno"] == "Recessione lunare", "generazioni"] = np.inf

    led.to_csv(outdir / "detectability.csv", index=False, float_format="%.6g")

    log("")
    log("  Registro di rilevabilita':")
    for _, r in led.iterrows():
        log(f"    {r['fenomeno']:20s} {r['osservabile'][:44]:44s} "
            f"{_fmt_years(r['anni_per_rilevare']):>8s} anni  "
            f"{'':2s}{r['generazioni']:>10.1f} gen"
            if np.isfinite(r["generazioni"]) else
            f"    {r['fenomeno']:20s} {r['osservabile'][:44]:44s} "
            f"{'mai':>8s}          --")

    # ---- LaTeX ------------------------------------------------------------
    tex = led.copy()
    tex["anni"] = tex["anni_per_rilevare"].map(_fmt_years)
    tex["gen"] = tex["generazioni"].map(
        lambda g: "--" if not np.isfinite(g) else f"{g:.1f}")
    tex["soglia_txt"] = tex.apply(
        lambda r: f"{r['soglia']:.0f}\\arcmin" if r["unita"] == "arcmin"
        else "1 giorno", axis=1)
    t1 = latex_table(
        tex, ["fenomeno", "osservabile", "soglia_txt", "anni", "gen"],
        ["Fenomeno", "Che cosa si osserva", "Soglia", "Anni", "Generazioni"],
        caption=(f"Rilevabilit\\`a a occhio nudo dei moti celesti, a "
                 f"{args.lat:.1f}$^\\circ$ di latitudine. Una generazione "
                 f"\\`e presa di {GENERATION_YEARS:.0f} anni. La soglia "
                 f"\\`e quella dello strumento con cui il fenomeno pu\\`o "
                 f"essere osservato, non la migliore possibile."),
        label="tab:rilevabilita",
        align="llrrr")

    hx = hor.copy()
    hx["lat"] = hx["lat_deg"].map(lambda v: f"{v:.1f}")
    hx["eq"] = hx["risoluzione_equinozio_giorni"].map(
        lambda v: "--" if not np.isfinite(v) else f"{v:.0f}")
    hx["ss"] = hx["risoluzione_solstizio_giorni"].map(
        lambda v: "--" if not np.isfinite(v) else f"{v:.0f}")
    hx["rp"] = hx["rapporto"].map(
        lambda v: "--" if not np.isfinite(v) else f"{v:.0f}")
    t2 = latex_table(
        hx, ["lat", "eq", "ss", "rp"],
        ["Latitudine", "Equinozio", "Solstizio", "Rapporto"],
        caption=("Giorni necessari perch\\'e il punto di levata del Sole si "
                 "sposti di un diametro solare (32\\arcmin). All'equinozio la "
                 "declinazione corre, al solstizio \\`e ferma: il solstizio "
                 "\\`e il marcatore meno preciso dei due, di un ordine di "
                 "grandezza."),
        label="tab:risoluzione-orizzonte",
        align="lrrr")

    (outdir / "tab_detectability.tex").write_text(t1, encoding="utf-8")
    (outdir / "tab_horizon_resolution.tex").write_text(t2, encoding="utf-8")

    # ---- figures -----------------------------------------------------------
    lats = np.arange(0.0, 66.0, 1.0)
    grid = np.full((lats.size, days.size), np.nan)
    for j, phi in enumerate(lats):
        _, _, r = date_resolution(float(phi), eps, thr_sun)
        grid[j] = r

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 9),
                                 gridspec_kw={"height_ratios": [2, 1]})
    im = a1.imshow(grid, origin="lower", aspect="auto",
                   extent=[0, days.size, lats[0], lats[-1]],
                   cmap="magma_r", vmin=0, vmax=30)
    fig.colorbar(im, ax=a1, label="giorni per un diametro solare")
    for x, nm in ((0, "eq. primavera"), (i_ss, "solstizio est."),
                  (int(round(TROPICAL_YEAR_D * 0.5)), "eq. autunno"),
                  (i_ws, "solstizio inv.")):
        a1.axvline(x, color="c", lw=0.8, ls="--")
        a1.text(x + 3, lats[-1] - 4, nm, color="c", fontsize=7, rotation=90,
                va="top")
    a1.set_ylabel("latitudine [°]")
    a1.set_title("Risoluzione di un calendario d'orizzonte\n"
                 "chiaro = la data si legge bene, scuro = il Sole non si "
                 "muove abbastanza da distinguere i giorni")

    a2.plot(days, res, lw=1.4, color="firebrick")
    a2.set_xlabel("giorno dell'anno dall'equinozio di primavera")
    a2.set_ylabel("giorni")
    a2.set_title(f"a {args.lat:.1f}° di latitudine")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "date_resolution.png", dpi=args.dpi,
                bbox_inches="tight")
    plt.close(fig)

    fin = led[np.isfinite(led["generazioni"])].sort_values("generazioni")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    y = np.arange(len(fin))
    ax.barh(y, fin["generazioni"], color="steelblue")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['fenomeno']}: {r['osservabile'][:42]}"
                        for _, r in fin.iterrows()], fontsize=8)
    ax.set_xscale("log")
    for g, lb, col in ((1, "una vita", "green"), (10, "dieci generazioni", "orange"),
                       (40, "un millennio", "red")):
        ax.axvline(g, color=col, lw=1.0, ls="--")
        ax.text(g, len(fin) - 0.4, lb, color=col, fontsize=7, rotation=90,
                va="top")
    ax.set_xlabel("generazioni prima che il cambiamento sia visibile "
                  f"(1 generazione = {GENERATION_YEARS:.0f} anni)")
    ax.set_title("Che cosa si accorge di cambiare, e dopo quanto\n"
                 "i fenomeni assenti dal grafico non diventano mai visibili")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "ledger.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    log("")
    log(f"Scritti {outdir}/detectability.csv, horizon_resolution.csv, "
        f"proper_motions.csv, tab_detectability.tex, "
        f"tab_horizon_resolution.tex, date_resolution.png, ledger.png")


if __name__ == "__main__":
    main()
