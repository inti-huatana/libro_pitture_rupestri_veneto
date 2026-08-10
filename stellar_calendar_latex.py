#!/usr/bin/env python3
"""
stellar_calendar_latex.py
Generate LaTeX appendix tables from stellar_calendar.py output CSVs.

Usage:
    Run stellar_calendar.py first to produce output/
    Then run this script to produce output/appendix_stellar_calendar.tex

Output: one longtable section per 10-kyr block (10 sections total, 0-10, 10-20, ..., 90-100 kyr BP)
Each section lists, per millennium, all heliacal risings sorted by day.
A separate table lists all discontinuities (appearances/disappearances).
"""

from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path("output")
TEX_OUT    = OUTPUT_DIR / "appendix_stellar_calendar.tex"

# Season abbreviations for compact table (nomi allineati a stellar_calendar.py:
# UNIVERSAL_WINDOWS + PALEO_WINDOWS + MESO_WINDOWS + FARM_WINDOWS)
SEASON_ABBR = {
    "Equinozio di primavera":                           "\\textsc{eq.pri}",
    "Solstizio d'estate":                               "\\textsc{sol.est}",
    "Equinozio d'autunno":                               "\\textsc{eq.aut}",
    "Solstizio d'inverno":                               "\\textsc{sol.inv}",
    "Disgelo dei fiumi":                                 "\\textsc{disgelo}",
    "Prime nevicate in pianura/collina":                 "\\textsc{prima neve}",
    "Nascita dei piccoli (stambecco, camoscio, cervo, bisonte, alce)": "\\textsc{nascite}",
    "Svernamento e parto dell'orso (letargo)":           "\\textsc{orso}",
    "Salita stagionale in quota":                        "\\textsc{salita}",
    "Discesa a valle (fine stagione d'alta quota)":      "\\textsc{discesa}",
    "Bramito del cervo":                                 "\\textsc{bramito}",
    "Amori del camoscio":                                "\\textsc{camoscio}",
    "Semina primaverile (legumi/miglio)":                "\\textsc{sem.pri}",
    "Semina autunnale (cereali)":                        "\\textsc{sem.aut}",
    "Mietitura":                                         "\\textsc{mietit.}",
    "":                                                  "---",
    "nan":                                               "---",
}

DISC_EVENT_IT = {
    "APPARE":            "appare",
    "SCOMPARE":          "scompare",
    "DIVENTA_CIRCUMPOLARE": "diventa circumpolare",
    "CESSA_CIRCUMPOLARE":   "cessa circumpolare",
    "SALTO_STAGIONALE":     "salto stagionale",
}


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}")


def tex_escape(s: str) -> str:
    return str(s).replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def vmag_fmt(v) -> str:
    try:
        return f"{float(v):.1f}"
    except (ValueError, TypeError):
        return "---"


def day_fmt(d) -> str:
    try:
        return f"{float(d):.0f}"
    except (ValueError, TypeError):
        return "---"


def build_heliacal_section(heliacal: pd.DataFrame, epoch_min: float, epoch_max: float) -> list[str]:
    """Build LaTeX lines for one 10-kyr block."""
    lines: list[str] = []
    block_label = f"{abs(int(epoch_max))}–{abs(int(epoch_min))} kyr BP" if epoch_min != epoch_max else f"{abs(int(epoch_min))} kyr BP"

    lines.append(f"\n\\subsection*{{Eventi eliaci {block_label}}}\n")
    lines.append("\\begin{small}")
    lines.append("\\begin{longtable}{rlllrl}")
    lines.append("\\toprule")
    lines.append("\\textbf{Epoca} & \\textbf{Periodo} & \\textbf{Stella} & \\textbf{V} & "
                 "\\textbf{Evento} & \\textbf{Stagione} \\\\")
    lines.append("\\midrule")
    lines.append("\\endfirsthead")
    lines.append("\\multicolumn{6}{c}{\\small\\textit{(continua)}} \\\\")
    lines.append("\\toprule")
    lines.append("\\textbf{Epoca} & \\textbf{Periodo} & \\textbf{Stella} & \\textbf{V} & "
                 "\\textbf{Evento} & \\textbf{Stagione} \\\\")
    lines.append("\\midrule")
    lines.append("\\endhead")
    lines.append("\\bottomrule")
    lines.append("\\endfoot")

    block = heliacal[
        (heliacal["epoch_kyr"] >= epoch_min) & (heliacal["epoch_kyr"] <= epoch_max)
    ].sort_values(["epoch_kyr", "giorno_evento"], ascending=[False, True])

    prev_epoch = None
    for _, row in block.iterrows():
        epoch = row["epoch_kyr"]
        if epoch != prev_epoch:
            if prev_epoch is not None:
                lines.append("\\midrule")
            epoch_label = f"{abs(int(epoch))} kyr BP" if epoch != 0 else "oggi"
            prev_epoch = epoch
        else:
            epoch_label = ""

        tipo = "levata" if row.get("tipo_evento") == "levata" else "tramonto"
        evento_abbr = SEASON_ABBR.get(str(row.get("evento", "nan")), "---")
        stella = row["star_label"]
        asterismo = str(row.get("asterismo", "") or "")
        if asterismo:
            stella = f"{stella} ({asterismo})"

        lines.append(
            f"{tex_escape(epoch_label)} & "
            f"{tex_escape(str(row.get('periodo', '')))} & "
            f"{tex_escape(stella)} & "
            f"{vmag_fmt(row['Vmag'])} & "
            f"{tipo} (g.\\,{day_fmt(row['giorno_evento'])}) & "
            f"{evento_abbr} \\\\"
        )

    lines.append("\\end{longtable}")
    lines.append("\\end{small}")
    return lines


def build_discontinuity_table(disc: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    lines.append("\n\\section*{Stelle che compaiono o scompaiono (0–100 kyr BP)}\n")
    lines.append("\\begin{small}")
    lines.append("\\begin{longtable}{rllrll}")
    lines.append("\\toprule")
    lines.append("\\textbf{Epoca} & \\textbf{Stella} & \\textbf{V} & "
                 "\\textbf{Evento} & \\textbf{Dettaglio} \\\\")
    lines.append("\\midrule")
    lines.append("\\endfirsthead")
    lines.append("\\multicolumn{5}{c}{\\small\\textit{(continua)}} \\\\")
    lines.append("\\midrule")
    lines.append("\\endhead")
    lines.append("\\bottomrule")
    lines.append("\\endfoot")

    # Only structurally significant events for this table
    sig_events = {"APPARE", "SCOMPARE", "DIVENTA_CIRCUMPOLARE", "CESSA_CIRCUMPOLARE"}
    sig = disc[disc["evento"].isin(sig_events)].sort_values(
        ["epoch_a_kyr", "Vmag"], ascending=[False, True]
    )

    for _, row in sig.iterrows():
        epoch_label = f"{abs(int(row['epoch_a_kyr']))} kyr BP" if row["epoch_a_kyr"] != 0 else "oggi"
        evento_it = DISC_EVENT_IT.get(str(row["evento"]), str(row["evento"]))
        lines.append(
            f"{tex_escape(epoch_label)} & "
            f"{tex_escape(row['star_label'])} & "
            f"{vmag_fmt(row['Vmag'])} & "
            f"{tex_escape(evento_it)} & "
            f"{tex_escape(str(row.get('dettaglio', '')))} \\\\"
        )

    lines.append("\\end{longtable}")
    lines.append("\\end{small}")
    return lines


def build_seasonal_shifts_table(disc: pd.DataFrame) -> list[str]:
    """Separate table for large seasonal shifts (>30 d) — potentially interesting for researchers."""
    lines: list[str] = []
    lines.append("\n\\section*{Spostamenti stagionali significativi ($>$30 giorni in 1 millennio)}\n")
    lines.append("\\begin{small}")
    lines.append("\\begin{longtable}{rllrl}")
    lines.append("\\toprule")
    lines.append("\\textbf{Epoca} & \\textbf{Stella} & \\textbf{V} & "
                 "\\textbf{Variazione} & \\textbf{Dettaglio} \\\\")
    lines.append("\\midrule")
    lines.append("\\endfirsthead")
    lines.append("\\multicolumn{5}{c}{\\small\\textit{(continua)}} \\\\")
    lines.append("\\midrule")
    lines.append("\\endhead")
    lines.append("\\bottomrule")
    lines.append("\\endfoot")

    shifts = disc[disc["evento"] == "SALTO_STAGIONALE"].sort_values(
        ["epoch_a_kyr", "Vmag"], ascending=[False, True]
    )
    for _, row in shifts.iterrows():
        epoch_label = f"{abs(int(row['epoch_a_kyr']))} kyr BP" if row["epoch_a_kyr"] != 0 else "oggi"
        lines.append(
            f"{tex_escape(epoch_label)} & "
            f"{tex_escape(row['star_label'])} & "
            f"{vmag_fmt(row['Vmag'])} & "
            f"salto $>$30 d & "
            f"{tex_escape(str(row.get('dettaglio', '')))} \\\\"
        )

    lines.append("\\end{longtable}")
    lines.append("\\end{small}")
    return lines


def build_preamble() -> list[str]:
    return [
        "% ============================================================",
        "% Appendice: Calendario stellare preistorico a 45°N",
        "% Generato automaticamente da stellar_calendar_latex.py",
        "% ============================================================",
        "",
        "\\chapter*{Appendice: Calendario stellare preistorico (0–100 kyr BP, 45°N)}",
        "\\addcontentsline{toc}{chapter}{Appendice: Calendario stellare preistorico}",
        "",
        "\\noindent",
        "Le tabelle seguenti elencano le stelle con magnitudine apparente $V < 3{,}0$",
        "visibili dalla latitudine $+45^\\circ$N (Veneto), organizzate per millennio.",
        "Il \\emph{giorno di levata eliacale} è espresso come numero di giorni dall'equinozio",
        "di primavera dell'epoca corrispondente (giorno~0).",
        "L'equinozio di primavera, il solstizio estivo (giorno~$\\approx$92),",
        "l'equinozio autunnale (giorno~$\\approx$183) e il solstizio invernale (giorno~$\\approx$274)",
        "sono calcolati con il modello di precessione di Vondrák et al.\\ (2011).",
        "Le magnitudini apparenti includono il moto proprio e la velocità radiale di ciascuna stella.",
        "",
        "\\medskip",
        "\\noindent\\textbf{Abbreviazioni stagionali:} finestre di $\\pm$10 giorni attorno al",
        "giorno indicato (0 = equinozio di primavera). Equinozi e solstizi valgono per",
        "tutte le epoche. \\textsc{disgelo}, \\textsc{prima neve}, \\textsc{nascite} e",
        "\\textsc{orso} si applicano solo al Paleolitico medio/superiore; \\textsc{salita},",
        "\\textsc{discesa}, \\textsc{bramito}, \\textsc{camoscio} e di nuovo \\textsc{orso}",
        "solo al Mesolitico; \\textsc{sem.pri}, \\textsc{sem.aut} e \\textsc{mietit.} solo",
        "dal Neolitico all'età del Ferro (vedi colonna Periodo). \\textsc{disgelo},",
        "\\textsc{prima neve}, \\textsc{salita} e \\textsc{discesa} si spostano inoltre di",
        "fase in fase climatica (Ultimo Massimo Glaciale, Tardoglaciale, Bølling-Allerød,",
        "Dryas Recente, Olocene): il giorno indicato qui è quello della fase più recente",
        "(analogo moderno/olocenico), non un valore fisso per tutto il periodo.",
        "\\textsc{eq.pri} Equinozio di primavera (g.~0);",
        "\\textsc{sol.est} Solstizio d'estate (g.~93);",
        "\\textsc{eq.aut} Equinozio d'autunno (g.~186);",
        "\\textsc{sol.inv} Solstizio d'inverno (g.~276);",
        "\\textsc{disgelo} Disgelo dei fiumi (g.~30, analogo moderno);",
        "\\textsc{prima neve} Prime nevicate in pianura/collina (g.~250, analogo moderno);",
        "\\textsc{nascite} Nascita dei piccoli, stambecco/camoscio/cervo/bisonte/alce (g.~85);",
        "\\textsc{orso} Svernamento e parto dell'orso, durante il letargo (g.~305);",
        "\\textsc{salita} Salita stagionale in quota (g.~65, analogo moderno);",
        "\\textsc{discesa} Discesa a valle, fine stagione d'alta quota (g.~225, analogo moderno);",
        "\\textsc{bramito} Bramito del cervo (g.~200);",
        "\\textsc{camoscio} Amori del camoscio (g.~250);",
        "\\textsc{sem.pri} Semina primaverile, legumi/miglio (g.~30);",
        "\\textsc{sem.aut} Semina autunnale, cereali (g.~215);",
        "\\textsc{mietit.} Mietitura (g.~100).",
        "",
        "\\bigskip",
    ]


def main() -> None:
    log("Loading CSVs ...")
    heliacal = pd.read_csv(OUTPUT_DIR / "heliacal_events.csv")
    disc     = pd.read_csv(OUTPUT_DIR / "discontinuities.csv")
    log(f"  heliacal: {len(heliacal):,} rows | discontinuities: {len(disc):,} rows")

    all_lines: list[str] = build_preamble()

    # One 10-kyr block per section, from 0 to -100
    block_edges = list(range(0, -101, -10))  # [0, -10, -20, ..., -100]
    for i in range(len(block_edges) - 1):
        e_max = float(block_edges[i])
        e_min = float(block_edges[i + 1])
        log(f"  Building section {e_min}–{e_max} kyr ...")
        all_lines.extend(build_heliacal_section(heliacal, e_min, e_max))

    log("Building discontinuity tables ...")
    all_lines.extend(build_discontinuity_table(disc))
    all_lines.extend(build_seasonal_shifts_table(disc))

    TEX_OUT.write_text("\n".join(all_lines), encoding="utf-8")
    log(f"Written → {TEX_OUT}  ({TEX_OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
