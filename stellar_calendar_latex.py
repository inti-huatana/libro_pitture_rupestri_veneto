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

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path("output")
TEX_OUT    = OUTPUT_DIR / "appendix_stellar_calendar.tex"

# Season abbreviations for compact table
SEASON_ABBR = {
    "Gelo/Inverno":       "\\textsc{inv}",
    "Disgelo/Primavera":  "\\textsc{pri}",
    "Semina/Pascolo":     "\\textsc{sem}",
    "Calura estiva":      "\\textsc{est}",
    "Raccolto":           "\\textsc{rac}",
    "Caccia autunnale":   "\\textsc{aut}",
    "Freddo/Pre-inverno": "\\textsc{fre}",
    "nan":                "---",
}

DISC_EVENT_IT = {
    "APPARE":            "appare",
    "SCOMPARE":          "scompare",
    "DIVENTA_CIRCUMPOLARE": "diventa circumpolare",
    "CESSA_CIRCUMPOLARE":   "cessa circumpolare",
    "SALTO_STAGIONALE":     "salto stagionale",
}


def log(msg: str) -> None:
    print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {msg}")


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

    lines.append(f"\n\\subsection*{{Levate eliache {block_label}}}\n")
    lines.append("\\begin{small}")
    lines.append("\\begin{longtable}{rllrrll}")
    lines.append("\\toprule")
    lines.append("\\textbf{Epoca} & \\textbf{Stella} & \\textbf{V} & "
                 "\\textbf{G. levata} & \\textbf{Stagione levata} & "
                 "\\textbf{G. tramonto} & \\textbf{Stagione tramonto} \\\\")
    lines.append("\\midrule")
    lines.append("\\endfirsthead")
    lines.append("\\multicolumn{7}{c}{\\small\\textit{(continua)}} \\\\")
    lines.append("\\toprule")
    lines.append("\\textbf{Epoca} & \\textbf{Stella} & \\textbf{V} & "
                 "\\textbf{G. levata} & \\textbf{Stagione levata} & "
                 "\\textbf{G. tramonto} & \\textbf{Stagione tramonto} \\\\")
    lines.append("\\midrule")
    lines.append("\\endhead")
    lines.append("\\bottomrule")
    lines.append("\\endfoot")

    block = heliacal[
        (heliacal["epoch_kyr"] >= epoch_min) & (heliacal["epoch_kyr"] <= epoch_max)
    ].sort_values(["epoch_kyr", "heliacal_rising_day"], ascending=[False, True])

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

        sr = SEASON_ABBR.get(str(row.get("season_rising", "nan")), "---")
        ss = SEASON_ABBR.get(str(row.get("season_setting", "nan")), "---")

        lines.append(
            f"{tex_escape(epoch_label)} & "
            f"{tex_escape(row['star_label'])} & "
            f"{vmag_fmt(row['Vmag'])} & "
            f"{day_fmt(row['heliacal_rising_day'])} & "
            f"{sr} & "
            f"{day_fmt(row['heliacal_setting_day'])} & "
            f"{ss} \\\\"
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
        "\\noindent\\textbf{Abbreviazioni stagionali:}",
        "\\textsc{inv} Gelo/Inverno (g.~315–45);",
        "\\textsc{pri} Disgelo/Primavera (g.~45–90);",
        "\\textsc{sem} Semina/Pascolo (g.~90–135);",
        "\\textsc{est} Calura estiva (g.~135–180);",
        "\\textsc{rac} Raccolto (g.~180–225);",
        "\\textsc{aut} Caccia autunnale (g.~225–270);",
        "\\textsc{fre} Freddo/Pre-inverno (g.~270–315).",
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
