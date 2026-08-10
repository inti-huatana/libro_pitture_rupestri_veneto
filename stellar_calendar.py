#!/usr/bin/env python3
"""
stellar_calendar.py
Process fullcat.csv.gz into prehistoric stellar calendar tables.

Usage:
    python3 stellar_calendar.py [options]

Options:
    --input     Path to fullcat.csv or fullcat.csv.gz  [default: auto-detect]
    --outdir    Output directory                        [default: output]
    --lat       Observer latitude in degrees            [default: 45.0]
    --Tmin      Minimum epoch in kyr (negative = past)  [default: -100.0]
    --Tmax      Maximum epoch in kyr                    [default: 0.0]
    --dt        Timestep in kyr                         [default: 1.0]
"""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd

# ── Asterism representatives (4 stars only) ─────────────────────────────────
ASTERISM_HIP: dict[int, str] = {
    17702: "Pleiadi",
    26727: "Cin.Orione",
    60718: "Cr.Sud",
    20894: "Iadi",
}

# ── Periodizzazione archeologica (anni espressi in kyr, 0 = oggi) ───────────
# Confini ripresi direttamente da chapters/14_glossario.tex (coerente con
# chapters/04_tempo_ghiacci_clima.tex, Ravazzi et al. 2007, per la
# terminologia "anni prima dell'anno zero astronomico" = epoch_kyr*1000).
# Due soglie (Paleolitico sup./Mesolitico e Mesolitico/Neolitico) sono il
# punto medio di un intervallo di transizione che la stessa fonte lascia
# sfumato (rispettivamente 12.700-11.000 e 8.000-7.500 anni prima
# dell'anno zero) invece di un confine netto: qui serve un singolo numero
# per assegnare ogni epoca a un periodo, quindi si usa il centro
# dell'intervallo, non un valore scelto a caso.
PERIODS: list[tuple[str, float, float]] = [
    ("Paleolitico medio",     -100.00,  -43.00),
    ("Paleolitico superiore",  -43.00,  -11.70),
    ("Mesolitico",             -11.70,   -7.75),
    ("Neolitico",                -7.75,   -5.25),
    ("Eneolitico (età del Rame)", -5.25,  -4.25),
    ("Età del Bronzo",           -4.25,   -2.75),
    ("Età del Ferro",            -2.75,    0.00),
]

# Periodi con economia di caccia-raccolta (nessuna semina/raccolta agricola)
# vs. periodi con economia agro-pastorale. Paleolitico e Mesolitico hanno
# ciascuno il proprio set di finestre (vedi PALEO_WINDOWS/MESO_WINDOWS
# più sotto): stesso modo di sussistenza, marcatori diversi.
PALEO_PERIODS = {"Paleolitico medio", "Paleolitico superiore"}
MESO_PERIODS = {"Mesolitico"}
FARMING_PERIODS = {"Neolitico", "Eneolitico (età del Rame)", "Età del Bronzo", "Età del Ferro"}


def _period_label(epoch_kyr) -> str:
    try:
        e = float(epoch_kyr)
    except (TypeError, ValueError):
        return ""
    for name, start, end in PERIODS:
        if start <= e <= end:
            return name
    return ""


# ── Finestre stagionali: (nome, giorno_centrale, mezza_ampiezza) ────────────
# giorno_centrale = giorni dall'equinozio di primavera (giorno 0).
# Mezza ampiezza = 10 giorni per tutte le finestre (non 5: un margine più
# realistico per pratiche datate solo approssimativamente in letteratura).
#
# Equinozi/solstizi: valori dalla lunghezza reale delle stagioni astronomiche
# (anno tropico, epoca J2000: primavera 92,8 gg, estate 93,6 gg, autunno
# 89,8 gg, inverno 89,0 gg — Meeus, "Astronomical Algorithms", 2a ed., 1998),
# non una divisione ingenua dell'anno in quarti uguali.
UNIVERSAL_WINDOWS: list[tuple[str, float, float]] = [
    ("Equinozio di primavera",  0.0, 10.0),
    ("Solstizio d'estate",     93.0, 10.0),
    ("Equinozio d'autunno",   186.0, 10.0),
    ("Solstizio d'inverno",   276.0, 10.0),
]

# Paleolitico (medio e superiore): nessuna semina/raccolta. Marcatori legati
# al clima glaciale/tardoglaciale e alla biologia della megafauna cacciata.
# Giorno 25 ≈ 14 aprile, giorno 85 ≈ 13 giugno, giorno 225 ≈ 31 ottobre.
# Cautela esplicita (nello stile del capitolo 4 del libro): il Paleolitico
# copre oscillazioni climatiche enormi — dal massimo glaciale (Ravazzi et
# al. 2007: estati più fredde di 8-10°C, linea delle nevi ~1.000-1.200 m
# più bassa) agli interstadi caldi (Bølling-Allerød) — quindi un singolo
# giorno fisso per "disgelo" e "prime nevicate" è un'approssimazione di
# primo ordine, non una data puntuale valida per l'intero periodo.
PALEO_WINDOWS: list[tuple[str, float, float]] = [
    # Disgelo dei fiumi: il disgelo primaverile è guidato soprattutto
    # dall'aumento dell'insolazione/fotoperiodo più che dalla temperatura
    # assoluta, quindi resta ancorato a ridosso dell'equinozio anche in
    # climi più freddi dell'attuale (v. cautela sopra).
    ("Disgelo dei fiumi",                      25.0, 10.0),
    # Prime nevicate in pianura/collina: in un clima mediamente più freddo
    # dell'attuale (Ravazzi et al. 2007), l'arrivo della neve a quote basse
    # è anticipato rispetto a oggi; qui si stima fine ottobre-inizio
    # novembre invece di novembre-dicembre.
    ("Prime nevicate in pianura/collina",     225.0, 10.0),
    # Nascita dei piccoli (stambecco, camoscio, cervo): stambecco
    # giugno-luglio, Gran Paradiso — Grignolio, Rossi, Bertolotto, Bassano &
    # Apollonio (2007), J. Wildlife Management 71(3) [Apollonio già in
    # bibliografia del libro per luccarini2006]; camoscio, tarda
    # primavera-inizio estate — Kourkgy et al. (2016), J. Animal Ecology;
    # cervo, giugno — dato coerente con georgii1981 (già in bibliografia)
    # sul periodo riproduttivo. Il capitolo 6 del libro descrive già la
    # caccia estiva ai "branchi di femmine con i loro piccoli" a Riparo
    # Soman (deangelis2021) — questa finestra formalizza in giorni quello
    # stesso fenomeno.
    ("Nascita dei piccoli (stambecco, camoscio, cervo)", 85.0, 10.0),
]

# Mesolitico: economia di caccia-raccolta come il Paleolitico, ma con un
# pattern insediativo diverso e meglio documentato — risalita stagionale
# verso siti d'alta quota. Fonte diretta per l'area di studio: Cima Dodici,
# Prealpi vicentine/Altopiano di Asiago, 2.000-2.100 m, frequentazione
# mesolitica antica stagionale (Peresani, Visentin et al. 2025, Quaternary
# International, "Highland settling in the Early Mesolithic. Insight from
# the record of Cima Dodici open-air sites, Venetian pre-Alps"). Bramito e
# amori sono ripresi dal Paleolitico: sono eventi fotoperiodici, quindi
# validi anche per il Mesolitico, e coerenti con la continuità della caccia
# a stambecco/camoscio nel Sauveterriano/Castelnoviano documentata nella
# stessa fonte.
MESO_WINDOWS: list[tuple[str, float, float]] = [
    ("Salita stagionale in quota",                   65.0, 10.0),
    ("Discesa a valle (fine stagione d'alta quota)", 225.0, 10.0),
    ("Bramito del cervo",                            200.0, 10.0),
    ("Amori del camoscio",                           250.0, 10.0),
]

# Comunità agro-pastorali (Neolitico → età del Ferro): calendario cerealicolo
# mediterraneo ad aridocoltura (semina autunnale prevalente, integrata da
# semina primaverile di leguminose/miglio; mietitura a ridosso del
# solstizio d'estate). Fonti: evidenza archeobotanica di semina sia autunnale
# sia primaverile già nel Neolitico europeo (infestanti dei campi coltivati);
# Plinio il Vecchio, Naturalis Historia XVIII (semina del grano "al tramonto
# delle Vergilie", cioè delle Pleiadi, fine ottobre-inizio novembre) e
# Columella, De Re Rustica II, per la continuità storica dello stesso
# calendario cerealicolo italico. Nota di cautela: Plinio/Columella
# descrivono l'Italia romana, non il Veneto neolitico; il dato è usato come
# indicazione di continuità climatico-colturale (stesse colture, stesso
# clima), non come prova diretta per il VI millennio prima dell'anno zero.
FARM_WINDOWS: list[tuple[str, float, float]] = [
    ("Semina primaverile (legumi/miglio)",  30.0, 10.0),
    ("Semina autunnale (cereali)",         215.0, 10.0),
    ("Mietitura",                          100.0, 10.0),
]


def _active_windows(epoch_kyr) -> list[tuple[str, float, float]]:
    """Finestre specifiche del periodo prima, universali dopo: se un evento
    stagionale specifico (es. nascita dei piccoli) cade nello stesso giorno
    di un equinozio/solstizio, vince l'evento più specifico."""
    period = _period_label(epoch_kyr)
    if period in PALEO_PERIODS:
        extra = PALEO_WINDOWS
    elif period in MESO_PERIODS:
        extra = MESO_WINDOWS
    elif period in FARMING_PERIODS:
        extra = FARM_WINDOWS
    else:
        extra = []
    return extra + UNIVERSAL_WINDOWS


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prehistoric stellar calendar from fullcat.csv.gz")
    p.add_argument("--input",  default=None,     help="Input CSV path")
    p.add_argument("--outdir", default="output",  help="Output directory")
    p.add_argument("--lat",    type=float, default=45.0,   help="Observer latitude in degrees")
    p.add_argument("--Tmin",   type=float, default=-100.0, help="Min epoch in kyr (200 or -200 both mean -200 kyr BP)")
    p.add_argument("--Tmax",   type=float, default=0.0,    help="Max epoch in kyr")
    p.add_argument("--dt",     type=float, default=1.0,    help="Timestep in kyr")
    args = p.parse_args()
    # Accept positive Tmin as meaning negative (years before present)
    if args.Tmin > args.Tmax:
        args.Tmin = -abs(args.Tmin)
    return args


def _star_label(row) -> str:
    name  = str(row.get("NAME",  "")).strip()
    bayer = str(row.get("Bayer", "")).strip()
    if name and name != "nan":
        label = name
    elif bayer and bayer != "nan":
        label = bayer
    else:
        label = f"HIP {int(row['HIP'])}"
    hip = int(row["HIP"])
    if hip in ASTERISM_HIP:
        label += f" ({ASTERISM_HIP[hip]})"
    return label


def _season_label(day, epoch_kyr) -> str:
    try:
        d = float(day) % 365.2422
    except (TypeError, ValueError):
        return ""
    for name, center, hw in _active_windows(epoch_kyr):
        lo = (center - hw) % 365.2422
        hi = (center + hw) % 365.2422
        if lo <= hi:
            if lo <= d <= hi:
                return name
        else:
            if d >= lo or d <= hi:
                return name
    return ""


def load_data(input_path: Path, lat_deg: float,
              tmin: float, tmax: float, dt: float) -> pd.DataFrame:
    log(f"Loading {input_path} ...")
    df = pd.read_csv(input_path, low_memory=False)
    log(f"  Raw rows: {len(df):,}")

    if "epoch_kyr_from_year0" in df.columns and "epoch_kyr" not in df.columns:
        df = df.rename(columns={"epoch_kyr_from_year0": "epoch_kyr"})

    df = df.drop_duplicates(subset=["HIP", "epoch_kyr"], keep="first")
    log(f"  After dedup: {len(df):,}")

    # Select epochs matching tmin..tmax at step dt
    n_steps = round((tmax - tmin) / dt)
    target_epochs = {round(tmin + i * dt, 6) for i in range(n_steps + 1)}
    df["_er"] = df["epoch_kyr"].round(6)
    df = df[df["_er"].isin(target_epochs)].drop(columns=["_er"])
    log(f"  After epoch filter [{tmin}, {tmax}] step {dt}: "
        f"{len(df):,} rows, {df['epoch_kyr'].nunique()} epochs")

    # Meridian altitude filter: 90 - |lat - dec| >= 2
    df = df.dropna(subset=["dec_deg"])
    alt = 90.0 - (lat_deg - df["dec_deg"]).abs()
    df = df[alt >= 2.0].copy()
    log(f"  After meridian alt >=2° filter: {len(df):,} rows")

    if df.empty:
        raise SystemExit("No data after filtering — check --Tmin/--Tmax against the file's epoch range.")
    df["star_label"] = df.apply(_star_label, axis=1)
    df["periodo"] = df["epoch_kyr"].apply(_period_label)
    return df.reset_index(drop=True)


def _pick_event(row) -> tuple[float, str, str]:
    """Per una stella/epoca: sceglie levata O tramonto (mai entrambi), quello
    che cade in una finestra stagionale attiva per il periodo dell'epoca.
    Se nessuno dei due cade in una finestra, usa comunque la levata come
    evento di riferimento (censimento generale), ma senza etichetta stagionale."""
    epoch = row["epoch_kyr"]
    rising = row["heliacal_rising_day"]
    setting = row.get("heliacal_setting_day", float("nan"))

    season_rising = _season_label(rising, epoch)
    if season_rising:
        return rising, "levata", season_rising

    season_setting = _season_label(setting, epoch)
    if season_setting:
        return setting, "tramonto", season_setting

    return rising, "levata", ""


def build_heliacal_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per epoch: top-2 brightest visible + 1 per asterismo (solo se il suo
    evento cade in una finestra stagionale attiva)."""
    vis = df[
        (df["visibility"] == "VISIBILE") &
        df["heliacal_rising_day"].notna()
    ].copy()
    vis["asterism"] = vis["HIP"].map(ASTERISM_HIP).fillna("")

    records: list[dict] = []
    for epoch, grp in vis.groupby("epoch_kyr", sort=True):
        grp_s = grp.sort_values("Vmag")
        top2 = grp_s.head(2)
        selected = set(top2["HIP"].tolist())

        extra: list[pd.Series] = []
        for hip in ASTERISM_HIP:
            if hip not in selected:
                rows = grp_s[grp_s["HIP"] == hip]
                if not rows.empty:
                    cand = rows.iloc[0]
                    _, _, season = _pick_event(cand)
                    if season:
                        extra.append(cand)
                        selected.add(hip)

        chosen = pd.concat([top2] + ([pd.DataFrame(extra)] if extra else []))
        for _, row in chosen.iterrows():
            day, tipo, season = _pick_event(row)
            records.append({
                "epoch_kyr":   row["epoch_kyr"],
                "periodo":     row["periodo"],
                "HIP":         int(row["HIP"]),
                "star_label":  row["star_label"],
                "asterism":    row["asterism"],
                "Vmag":        row["Vmag"],
                "giorno_evento": day,
                "tipo_evento":   tipo,
                "stagione":      season,
            })

    out = pd.DataFrame(records)
    if not out.empty:
        out = out.sort_values(["epoch_kyr", "Vmag"])
    log(f"  Heliacal events: {len(out):,} rows")
    return out


def build_seasonal_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Una riga per epoca, una colonna per ciascun evento stagionale
    (equinozi/solstizi + eventi specifici del periodo). Cella = stella più
    brillante il cui evento (levata o tramonto) cade in quella finestra,
    con la sua magnitudine."""
    vis = df[
        (df["visibility"] == "VISIBILE") &
        df["heliacal_rising_day"].notna()
    ].copy()

    all_event_names = [w[0] for w in UNIVERSAL_WINDOWS] + \
                       [w[0] for w in PALEO_WINDOWS] + \
                       [w[0] for w in MESO_WINDOWS] + \
                       [w[0] for w in FARM_WINDOWS]
    # Deduplica preservando l'ordine di prima comparsa, nel caso in cui una
    # revisione futura riusi lo stesso nome di evento in più liste (oggi non
    # succede: nessun nome è condiviso tra le liste).
    seen: set[str] = set()
    all_event_names = [n for n in all_event_names if not (n in seen or seen.add(n))]

    rows: list[dict] = []
    for epoch, grp in vis.groupby("epoch_kyr", sort=True):
        row = {"epoch_kyr": epoch, "periodo": _period_label(epoch)}
        active_names = {w[0] for w in _active_windows(epoch)}
        best: dict[str, tuple[float, str]] = {}  # event_name -> (Vmag, cell text)

        for _, star in grp.iterrows():
            _, _, season = _pick_event(star)
            if not season:
                continue
            vmag = star["Vmag"]
            if season not in best or vmag < best[season][0]:
                best[season] = (vmag, f"{star['star_label']} ({vmag:.1f})")

        for name in all_event_names:
            if name in active_names and name in best:
                row[name] = best[name][1]
            else:
                row[name] = ""
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("epoch_kyr")
    log(f"  Seasonal calendar: {len(out):,} rows, {len(all_event_names)} event columns")
    return out


def build_discontinuities(df: pd.DataFrame) -> pd.DataFrame:
    """Detect appearances, disappearances, circumpolar transitions."""
    d = (df.groupby(["HIP", "epoch_kyr"], sort=False)
           .first().reset_index()
           .sort_values(["HIP", "epoch_kyr"], ascending=[True, False])
           .copy())
    d["visibility"] = d["visibility"].fillna("ASSENTE")

    grp = d.groupby("HIP", sort=False)
    d["vis_prev"]   = grp["visibility"].shift(-1).fillna("ASSENTE")
    d["rise_prev"]  = grp["heliacal_rising_day"].shift(-1)
    d["epoch_prev"] = grp["epoch_kyr"].shift(-1)
    d = d.dropna(subset=["epoch_prev"]).copy()

    vis  = d["visibility"]
    visp = d["vis_prev"]

    event_masks: dict[str, pd.Series] = {
        "APPARE":               (visp == "ASSENTE") & (vis == "VISIBILE"),
        "SCOMPARE":             (visp == "VISIBILE") & (vis == "ASSENTE"),
        "DIVENTA_CIRCUMPOLARE": (visp != "CIRCUMPOLARE") & (vis == "CIRCUMPOLARE"),
        "CESSA_CIRCUMPOLARE":   (visp == "CIRCUMPOLARE") & (vis != "CIRCUMPOLARE"),
        "SALTO_STAGIONALE": (
            (vis == "VISIBILE") & (visp == "VISIBILE") &
            d["heliacal_rising_day"].notna() & d["rise_prev"].notna() &
            ((d["heliacal_rising_day"] - d["rise_prev"]).abs() > 30)
        ),
    }

    parts: list[pd.DataFrame] = []
    for evento, mask in event_masks.items():
        sub = d[mask].copy()
        sub["evento"]      = evento
        sub["epoch_a_kyr"] = sub["epoch_kyr"]
        if evento == "SALTO_STAGIONALE":
            sub["dettaglio"] = (
                "g." + sub["heliacal_rising_day"].round(0).astype(int).astype(str) +
                " <- g." + sub["rise_prev"].round(0).astype(int).astype(str)
            )
        else:
            sub["dettaglio"] = ""
        parts.append(sub[["epoch_a_kyr", "periodo", "HIP", "star_label", "Vmag", "evento", "dettaglio"]])

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["epoch_a_kyr", "Vmag"], ascending=[False, True])
    log(f"  Discontinuities: {len(out):,} rows")
    return out


def build_pole_star_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per ogni epoca: le 3 stelle circumpolari più vicine al polo (per
    distanza angolare), elencate dalla più brillante alla meno brillante
    (ordine per Vmag crescente). Nessun punteggio combinato."""
    circ = df[df["visibility"] == "CIRCUMPOLARE"].dropna(subset=["dec_deg", "Vmag"]).copy()
    circ["pole_dist_deg"] = 90.0 - circ["dec_deg"]
    circ = circ[circ["pole_dist_deg"] >= 0]

    records: list[dict] = []
    for epoch, grp in circ.groupby("epoch_kyr", sort=True):
        nearest3 = grp.sort_values("pole_dist_deg").head(3)
        nearest3 = nearest3.sort_values("Vmag")
        for rank, (_, star) in enumerate(nearest3.iterrows(), start=1):
            records.append({
                "epoch_kyr":     epoch,
                "periodo":       star["periodo"],
                "rank":          rank,
                "HIP":           int(star["HIP"]),
                "star_label":    star["star_label"],
                "Vmag":          star["Vmag"],
                "dec_deg":       star["dec_deg"],
                "pole_dist_deg": star["pole_dist_deg"],
            })

    out = pd.DataFrame(records)
    log(f"  Pole star table: {out['epoch_kyr'].nunique() if not out.empty else 0} epochs, {len(out):,} rows")
    return out


def build_epoch_summary(df: pd.DataFrame, heliacal: pd.DataFrame) -> pd.DataFrame:
    vis_count  = (df[df["visibility"] == "VISIBILE"]
                  .groupby("epoch_kyr")["HIP"].nunique()
                  .rename("n_visible"))
    circ_count = (df[df["visibility"] == "CIRCUMPOLARE"]
                  .groupby("epoch_kyr")["HIP"].nunique()
                  .rename("n_circumpolare"))
    hel_count  = (heliacal.groupby("epoch_kyr")["HIP"].nunique()
                  .rename("n_heliacal") if not heliacal.empty
                  else pd.Series(dtype=int, name="n_heliacal"))

    out = (pd.concat([vis_count, circ_count, hel_count], axis=1)
             .fillna(0).astype(int)
             .reset_index()
             .sort_values("epoch_kyr", ascending=False))
    out["periodo"] = out["epoch_kyr"].apply(_period_label)
    log(f"  Epoch summary: {len(out):,} epochs")
    return out


def main() -> None:
    args = parse_args()

    if args.input:
        input_path = Path(args.input)
    else:
        input_path = (Path("fullcat.csv.gz") if Path("fullcat.csv.gz").exists()
                      else Path("fullcat.csv"))

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path, args.lat, args.Tmin, args.Tmax, args.dt)

    log("Building heliacal table ...")
    heliacal = build_heliacal_table(df)

    log("Building seasonal calendar ...")
    seasonal = build_seasonal_calendar(df)

    log("Building discontinuities ...")
    disc = build_discontinuities(df)

    log("Building pole star table ...")
    poles = build_pole_star_table(df)

    log("Building epoch summary ...")
    summary = build_epoch_summary(df, heliacal)

    heliacal.to_csv(out_dir / "heliacal_events.csv",   index=False)
    seasonal.to_csv(out_dir / "seasonal_calendar.csv",  index=False)
    disc.to_csv(    out_dir / "discontinuities.csv",    index=False)
    poles.to_csv(   out_dir / "pole_stars.csv",         index=False)
    summary.to_csv( out_dir / "epoch_summary.csv",      index=False)

    log(f"Done. Output in {out_dir}/")
    log(f"  heliacal_events.csv    {len(heliacal):,} rows")
    log(f"  seasonal_calendar.csv  {len(seasonal):,} rows")
    log(f"  discontinuities.csv    {len(disc):,} rows")
    log(f"  pole_stars.csv         {len(poles):,} rows")
    log(f"  epoch_summary.csv      {len(summary):,} rows")


if __name__ == "__main__":
    main()
