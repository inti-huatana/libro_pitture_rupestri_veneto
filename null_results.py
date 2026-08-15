#!/usr/bin/env python3
"""
null_results.py

What did not change, and the numbers that say so.

A result that comes out zero is still a result, provided the zero is measured
rather than assumed. Everything else in this set is a drift; this program is
the other column of the ledger, and it exists because several things a reader
would expect to have varied over two hundred millennia demonstrably did not,
and because a few that did are too small to have ever been seen.

Each entry is one comparison: the size of the change over the span, against the
threshold of the instrument that could have registered it. The ratio of the two
is the whole verdict. Below one, the phenomenon never happened as far as anyone
could tell; above one, it belongs in the other ledger.

Three of these are worth stating in advance.

Nutation is invisible and always was. Its principal term is 9.2 arcseconds,
against a naked-eye threshold near one arcminute: short by a factor of six, and
undetectable before the telescope. But the same 18.6-year cycle drives the
lunar standstills, whose amplitude at the horizon is some sixteen degrees --
three orders of magnitude larger, and the thing whole monuments were built to
catch. One period, two manifestations, and the reason to compute both here is
that stating the ratio between them is more useful than dismissing the first.

The Moon was the same Moon. It recedes by 3.8 cm a year, so over 200 kyr it
gained 7.6 km on a distance of 384 400: two parts in a hundred thousand. Its
apparent diameter changed by four hundredths of an arcsecond, and the balance
between total and annular eclipses is indistinguishable from today's. The near
equality of the solar and lunar discs, which is a coincidence of our geological
moment on a scale of billions of years, held rigorously fixed for the whole of
human prehistory.

The planets kept the same time. They have no tides worth the name, so their
periods in seconds are constant, and the day and the year stretch together, so
their periods in days scale identically and the commensurabilities between them
are invariant. Venus's five synodic periods against eight years came out the
same, to a fraction of a second, 200 kyr ago as now. Which is the sharpest
contrast the whole project can offer: the pole star changed, the constellations
came apart, the heliacal risings walked through the seasons, the galactic
centre rose and fell through forty-seven degrees -- and Venus did exactly what
it does now.

The lunisolar cycles are the one place where a real change hides, and it is
reported with its assumption showing. The Metonic residual is today about two
hours in nineteen years. Running the tidal recession backwards at a constant
rate makes that residual pass through zero some tens of millennia ago and
reverse sign, which would mean the cycle was once exact. That conclusion is
only as good as the constant rate, and the rate is not constant: the present
value is anomalously high because the ocean basins are near a tidal resonance,
and the long-term average was lower. So the epoch of exactness is quoted, and
so is its sensitivity to the assumption, and the reader is told not to trust
the first without the second.

Usage
-----
    python3 null_results.py --span 200 --outdir nulls
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

# ---- thresholds, arcminutes unless stated ---------------------------------
THR_VERNIER_ARCMIN = 1.0        # relative position, the eye at its best
THR_ABSOLUTE_ARCMIN = 10.0      # against a horizon, floored by refraction
THR_DISC_ARCMIN = 32.0          # the solar and lunar discs, the field rulers
THR_DAY = 1.0                   # a calendar kept by counting

# ---- present values, SI ----------------------------------------------------
DAY_S = 86400.0
YEAR_SID_D = 365.256363
YEAR_TROP_D = 365.2422
MONTH_SID_D = 27.321662
MONTH_SYN_D = 29.530589
MONTH_DRAC_D = 27.212221
NODE_PERIOD_D = 6798.38

MOON_A_M = 3.844e8
MOON_DIAM_ARCMIN = 31.1
SUN_DIAM_ARCMIN = 32.0

LUNAR_INCLINATION_DEG = 5.145
NUTATION_OBLIQ_ARCSEC = 9.2
NUTATION_LON_ARCSEC = 17.2
NUTATION_PERIOD_YR = 18.61

ABERRATION_ARCSEC = 20.5        # annual aberration constant
PARALLAX_MAX_ARCSEC = 0.768     # Proxima, the largest there is

VENUS_SYNODIC_D = 583.92
JUPITER_SYNODIC_D = 398.88
SATURN_SYNODIC_D = 378.09

# Rates whose value carries an assumption, exposed on the command line.
RECESSION_M_PER_YR = 0.038      # present lunar recession
DAY_LENGTHENING_S_PER_YR = 1.7e-5   # 1.7 ms per century


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


def rise_azimuth_deg(dec_deg, lat_deg: float):
    c = math.sin(math.radians(dec_deg)) / math.cos(math.radians(lat_deg))
    return math.degrees(math.acos(c)) if abs(c) <= 1.0 else float("nan")


def precession_rate_arcsec_per_yr(epochs_kyr: np.ndarray) -> np.ndarray:
    """General precession in longitude at each epoch, from the same model.

    Recovered from the pole's own speed divided by sin(eps), so that it and the
    obliquity used elsewhere cannot drift apart.
    """
    out = np.empty(epochs_kyr.size)
    step = 0.05
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, t in enumerate(epochs_kyr):
            pa = np.asarray(erfa.ltpequ(1000.0 * (t - 0.5 * step)), dtype=float)
            pb = np.asarray(erfa.ltpequ(1000.0 * (t + 0.5 * step)), dtype=float)
            ea = np.asarray(erfa.ltpecl(1000.0 * t), dtype=float)
            eq = np.asarray(erfa.ltpequ(1000.0 * t), dtype=float)
            pole = math.degrees(math.acos(
                np.clip(float(np.dot(pa, pb)), -1.0, 1.0))) * 3600.0 / (1000.0 * step)
            eps = math.acos(np.clip(float(np.dot(eq, ea)), -1.0, 1.0))
            out[i] = pole / math.sin(eps)
    return out


def lunisolar_series(epochs_kyr: np.ndarray, recession: float,
                     day_rate: float):
    """Month, year and day at each epoch, and the residuals of the two cycles.

    Everything is carried in SI seconds first and converted to days last,
    because that is where the two effects separate: tides lengthen the month in
    seconds, and quite independently they lengthen the day, so a period
    expressed in days moves for two reasons that partly cancel. The scalings
    are the classical first-order ones -- the sidereal month goes as a to the
    three halves by Kepler, and the nodal period as a to the minus three halves,
    since the solar perturbation that drives the regression scales with the
    square of the ratio of mean motions.
    """
    years = 1000.0 * epochs_kyr
    a = MOON_A_M + recession * years
    ratio = a / MOON_A_M

    day_s = DAY_S + day_rate * years
    sid_s = MONTH_SID_D * DAY_S * ratio ** 1.5
    node_s = NODE_PERIOD_D * DAY_S * ratio ** -1.5

    year_sid_s = YEAR_SID_D * DAY_S                 # no tides on the Earth's orbit
    p = precession_rate_arcsec_per_yr(epochs_kyr)
    year_trop_s = year_sid_s * (1.0 - p / 1296000.0)

    syn_s = 1.0 / (1.0 / sid_s - 1.0 / year_sid_s)
    drac_s = 1.0 / (1.0 / sid_s + 1.0 / node_s)

    month_syn_d = syn_s / day_s
    month_drac_d = drac_s / day_s
    year_trop_d = year_trop_s / day_s

    return pd.DataFrame({
        "epoch_kyr": epochs_kyr,
        "day_s": day_s,
        "moon_a_m": a,
        "month_synodic_d": month_syn_d,
        "month_draconic_d": month_drac_d,
        "year_tropical_d": year_trop_d,
        "precession_arcsec_per_yr": p,
        "metonic_residual_d": 235.0 * month_syn_d - 19.0 * year_trop_d,
        "saros_residual_d": 223.0 * month_syn_d - 242.0 * month_drac_d,
        "venus_8yr_residual_d": 8.0 * year_trop_d - 5.0 * (
            VENUS_SYNODIC_D * DAY_S / day_s),
    })


def zero_crossing(x: np.ndarray, y: np.ndarray):
    """Where a series changes sign, by linear interpolation; NaN if it does not."""
    s = np.sign(y)
    k = np.flatnonzero(s[:-1] * s[1:] < 0)
    if k.size == 0:
        return np.nan
    i = int(k[0])
    return float(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))


def main() -> None:
    p = argparse.ArgumentParser(
        description="The quantities that did not change, with the numbers.")
    p.add_argument("--span", type=float, default=200.0,
                   help="half-span considered, kyr")
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--lat", type=float, default=45.5)
    p.add_argument("--recession", type=float, default=RECESSION_M_PER_YR,
                   help="lunar recession, m/yr; the present value is "
                        "anomalously high")
    p.add_argument("--day-rate", type=float, default=DAY_LENGTHENING_S_PER_YR,
                   help="lengthening of the day, s/yr")
    p.add_argument("--outdir", default="nulls")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    span_yr = 1000.0 * args.span
    epochs = np.arange(-args.span, args.span + 0.5 * args.dt, args.dt)

    log(f"Intervallo +/- {args.span:.0f} kyr, latitudine {args.lat:.1f}°")
    log(f"Recessione lunare {args.recession * 100:.2f} cm/anno, "
        f"allungamento del giorno {args.day_rate * 1e5:.2f} ms/secolo")

    ls = lunisolar_series(epochs, args.recession, args.day_rate)
    ls.to_csv(outdir / "lunisolar_series.csv", index=False,
              float_format="%.9f")
    now = int(np.argmin(np.abs(epochs)))

    # ---- the ledger of nulls ----------------------------------------------
    d_moon = args.recession * span_yr
    d_moon_diam = MOON_DIAM_ARCMIN * (d_moon / MOON_A_M)
    d_day = args.day_rate * span_yr
    d_year_days = YEAR_TROP_D * (d_day / DAY_S)

    az_lo = rise_azimuth_deg(ls["year_tropical_d"].iloc[now] * 0 + 23.44
                             - LUNAR_INCLINATION_DEG, args.lat)
    az_hi = rise_azimuth_deg(23.44 + LUNAR_INCLINATION_DEG, args.lat)
    standstill_arcmin = abs(az_hi - az_lo) * 60.0

    ptr = ls["precession_arcsec_per_yr"]
    d_year_prec_s = (YEAR_SID_D * DAY_S
                     * (ptr.max() - ptr.min()) / 1296000.0)

    entries = [
        ("Nutazione in obliquita'", "posizione del polo fra le stelle",
         NUTATION_OBLIQ_ARCSEC / 60.0, "arcmin", THR_VERNIER_ARCMIN),
        ("Nutazione in longitudine", "posizione del polo fra le stelle",
         NUTATION_LON_ARCSEC / 60.0, "arcmin", THR_VERNIER_ARCMIN),
        ("Lunistizio maggiore", f"azimut di levata della Luna a {args.lat:.1f}°",
         standstill_arcmin, "arcmin", THR_DISC_ARCMIN),
        ("Aberrazione annua", "posizione di una stella",
         ABERRATION_ARCSEC / 60.0, "arcmin", THR_VERNIER_ARCMIN),
        ("Parallasse stellare", "posizione della stella piu' vicina",
         2.0 * PARALLAX_MAX_ARCSEC / 60.0, "arcmin", THR_VERNIER_ARCMIN),
        ("Recessione lunare", "diametro apparente della Luna",
         d_moon_diam, "arcmin", THR_DISC_ARCMIN),
        ("Allungamento del giorno", "giorni contati in un anno",
         d_year_days, "giorni", THR_DAY),
        ("Precessione variabile", "durata dell'anno tropico",
         d_year_prec_s / DAY_S, "giorni", THR_DAY),
        ("Eccentricita' massima", "variazione annua del disco solare",
         2.0 * 0.06 * SUN_DIAM_ARCMIN, "arcmin", THR_DISC_ARCMIN),
        ("Periodo sinodico di Venere", "durata del ciclo in giorni",
         VENUS_SYNODIC_D * (d_day / DAY_S), "giorni", THR_DAY),
        ("Periodo sinodico di Giove", "durata del ciclo in giorni",
         JUPITER_SYNODIC_D * (d_day / DAY_S), "giorni", THR_DAY),
        ("Periodo sinodico di Saturno", "durata del ciclo in giorni",
         SATURN_SYNODIC_D * (d_day / DAY_S), "giorni", THR_DAY),
        ("Ciclo metonico", "residuo su 19 anni",
         abs(ls["metonic_residual_d"].max() - ls["metonic_residual_d"].min()),
         "giorni", THR_DAY),
        ("Saros", "residuo su 18 anni",
         abs(ls["saros_residual_d"].max() - ls["saros_residual_d"].min()),
         "giorni", THR_DAY),
        ("Ciclo di 8 anni di Venere", "residuo su 8 anni",
         abs(ls["venus_8yr_residual_d"].max()
             - ls["venus_8yr_residual_d"].min()), "giorni", THR_DAY),
    ]

    rows = []
    for what, observable, change, unit, thr in entries:
        rows.append({"quantita": what, "osservabile": observable,
                     "variazione": change, "unita": unit, "soglia": thr,
                     "rapporto": change / thr,
                     "verdetto": "visibile" if change >= thr
                     else "mai visibile"})
    nul = pd.DataFrame(rows).sort_values("rapporto")
    nul.to_csv(outdir / "null_results.csv", index=False, float_format="%.6g")

    log("")
    log(f"  Variazione su +/- {args.span:.0f} kyr, contro la soglia "
        f"dell'occhio:")
    for _, r in nul.iterrows():
        log(f"    {r['quantita']:28s} {r['variazione']:11.4g} "
            f"{r['unita']:7s} / soglia {r['soglia']:5.1f}  = "
            f"{r['rapporto']:10.3g}   {r['verdetto']}")

    n_null = int((nul["rapporto"] < 1.0).sum())
    log("")
    log(f"  {n_null} voci su {len(nul)} restano sotto soglia: non sono mai "
        f"state osservabili, in tutta la preistoria umana.")

    r_nut = float(nul.loc[nul["quantita"] == "Nutazione in obliquita'",
                          "variazione"].iloc[0])
    r_std = float(nul.loc[nul["quantita"] == "Lunistizio maggiore",
                          "variazione"].iloc[0])
    log(f"  Nutazione e lunistizio hanno lo stesso periodo di "
        f"{NUTATION_PERIOD_YR:.2f} anni e ampiezze in rapporto "
        f"{r_std / r_nut:,.0f} a 1.")

    # ---- the lunisolar cycles ---------------------------------------------
    log("")
    log("  Cicli lunisolari:")
    for col, n, label in (("metonic_residual_d", 19, "Metone (19 anni)"),
                          ("saros_residual_d", 18, "Saros (223 lunazioni)"),
                          ("venus_8yr_residual_d", 8, "Venere (8 anni)")):
        y = ls[col].to_numpy()
        z = zero_crossing(epochs, y)
        log(f"    {label:24s} adesso {y[now]:+9.5f} giorni, "
            f"escursione {y.max() - y.min():.5f} giorni"
            + (f", si annulla a {z:+.1f} kyr" if np.isfinite(z) else
               ", non si annulla mai"))

    # Sensitivity of that crossing to the assumption it rests on.
    log("")
    log("  Sensibilita' dell'epoca di esattezza del ciclo metonico al tasso "
        "di recessione assunto:")
    sens = []
    for f in (0.4, 0.6, 0.8, 1.0, 1.2):
        s = lunisolar_series(epochs, args.recession * f, args.day_rate * f)
        z = zero_crossing(epochs, s["metonic_residual_d"].to_numpy())
        sens.append({"fattore_tasso": f,
                     "recessione_cm_per_anno": 100.0 * args.recession * f,
                     "metone_zero_kyr": z})
        log(f"    tasso x{f:.1f} ({100 * args.recession * f:.2f} cm/anno)  ->  "
            + (f"{z:+.1f} kyr" if np.isfinite(z) else "mai"))
    pd.DataFrame(sens).to_csv(outdir / "metonic_sensitivity.csv", index=False,
                              float_format="%.4f")
    log("    il tasso attuale e' anomalo per risonanza dei bacini oceanici: "
        "la media di lungo periodo e' minore, quindi l'epoca vera e' piu' "
        "remota di quella nominale.")

    # ---- figures -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, max(5, 0.42 * len(nul))))
    y = np.arange(len(nul))
    col = ["firebrick" if r >= 1 else "steelblue" for r in nul["rapporto"]]
    ax.barh(y, nul["rapporto"], color=col)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['quantita']}: {r['osservabile'][:38]}"
                        for _, r in nul.iterrows()], fontsize=8)
    ax.set_xscale("log")
    ax.axvline(1.0, color="k", lw=1.2)
    ax.set_xlabel("variazione / soglia di rilevabilita'  "
                  f"(su +/- {args.span:.0f} kyr)")
    ax.set_title("Che cosa non e' cambiato\n"
                 "a sinistra della linea: mai osservabile in tutta la "
                 "preistoria")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "null_results.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    a1.plot(epochs, ls["metonic_residual_d"], lw=1.4, color="darkorange")
    a1.axhline(0.0, color="k", lw=0.8)
    a1.set_ylabel("giorni")
    a1.set_title("Residuo del ciclo metonico: 235 lunazioni meno 19 anni\n"
                 "a tasso di dissipazione costante, che e' l'ipotesi debole")
    a1.grid(alpha=0.3)

    a2.plot(epochs, ls["saros_residual_d"], lw=1.4, color="teal")
    a2.axhline(0.0, color="k", lw=0.8)
    a2.set_ylabel("giorni")
    a2.set_title("Residuo del Saros: 223 mesi sinodici meno 242 draconici")
    a2.grid(alpha=0.3)

    a3.plot(epochs, ls["venus_8yr_residual_d"]
            - ls["venus_8yr_residual_d"].iloc[now], lw=1.4, color="navy")
    a3.axhline(0.0, color="k", lw=0.8)
    a3.set_xlabel("epoca [kyr dall'anno 0]")
    a3.set_ylabel("giorni, scarto da oggi")
    a3.set_title("Residuo del ciclo di 8 anni di Venere, scarto dal valore "
                 "attuale\nil giorno e l'anno si allungano insieme, "
                 "quindi il rapporto non si muove")
    a3.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "lunisolar_cycles.png", dpi=args.dpi,
                bbox_inches="tight")
    plt.close(fig)

    log("")
    log(f"Scritti {outdir}/null_results.csv, lunisolar_series.csv, "
        f"metonic_sensitivity.csv, null_results.png, lunisolar_cycles.png")


if __name__ == "__main__":
    main()
