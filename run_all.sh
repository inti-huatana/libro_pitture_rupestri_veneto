#!/usr/bin/env bash
#
# run_all.sh -- l'intera catena, in ordine, dai 100 kyr passati a oggi.
#
# I vicoli ciechi (constellation_age.py, bright_star_longterm.py,
# pole_stars_grid.py) non compaiono: vedi README_programmi.md.
#
# Uso:
#     ./run_all.sh                     tutto
#     ./run_all.sh 3 4 5               solo le fasi indicate
#     DRY=1 ./run_all.sh               stampa i comandi senza eseguirli
#
set -euo pipefail

# ---------------------------------------------------------------- parametri
CAT=${CAT:-hip_mag65.csv}       # catalogo Hipparcos di ingresso, Hp < 6.5
SKY=${SKY:-skycultures}         # albero delle culture celesti di Stellarium
OUT=${OUT:-out}                 # radice degli output

TMIN=${TMIN:--100}              # epoca piu' antica, kyr dall'anno 0
TMAX=${TMAX:-0}
DT=${DT:-0.5}                   # passo temporale, kyr

LAT=${LAT:-45.5}                # il Veneto
LAT_MIN=${LAT_MIN:--60}
LAT_MAX=${LAT_MAX:-70}
LAT_STEP=${LAT_STEP:-5}

# Oltre la quarta magnitudine una stella non si distingue dalle vicine: e' il
# limite del riconoscimento, non quello della visione. Il catalogo arriva a
# 6.5, quindi il taglio va imposto e non ereditato.
VMAX=${VMAX:-4.0}

CULTURE=${CULTURE:-modern_iau}  # per constellation_drift.py
WORKERS=${WORKERS:-$(nproc 2>/dev/null || echo 4)}

GRID=$OUT/grid                  # traiettorie, catalogo completo
GRIDB=$OUT/grid_bright          # traiettorie + fullcat, sottoinsieme V<VMAX
SKYP=$OUT/skycultures_parsed

# ------------------------------------------------------------------ utility
say() { printf '\n\033[1m=== %s\033[0m\n' "$*"; }
run() {
    printf '\033[2m$ %s\033[0m\n' "$*"
    [[ ${DRY:-0} == 1 ]] || "$@"
}
want() {                        # want <n>  ->  la fase n va eseguita?
    [[ $# -gt 0 ]] || return 0
    (( $# )) && { [[ ${#PHASES[@]} -eq 0 ]] && return 0
                  for p in "${PHASES[@]}"; do [[ $p == "$1" ]] && return 0; done
                  return 1; }
}
PHASES=("$@")

mkdir -p "$OUT"

# =========================================================================
# 0. generazione
# =========================================================================
# Due passaggi del motore, e la ragione e' il volume. trajectories.csv non
# dipende dalla latitudine e costa stelle x epoche; fullcat.csv costa
# stelle x epoche x latitudini, cioe' ventisette volte tanto, e con novemila
# stelle non sta su un disco. I soli programmi che leggono fullcat tagliano
# comunque a magnitudine bassa, quindi il secondo passaggio gira sul
# sottoinsieme riconoscibile e il primo salta fullcat del tutto.
if want 0; then
    say "0a. traiettorie, catalogo completo"
    run python3 bright_star_grid.py \
        --input "$CAT" --outdir "$GRID" \
        --Tmin "$TMIN" --Tmax "$TMAX" --dt "$DT" \
        --lat-min "$LAT_MIN" --lat-max "$LAT_MAX" --lat-step "$LAT_STEP" \
        --workers "$WORKERS" --no-fullcat

    say "0b. sottoinsieme V<$VMAX, con fullcat"
    run python3 -c "
import sys, pandas as pd
d = pd.read_csv(sys.argv[1])
n0 = len(d)
d = d[d['Vmag'] <= float(sys.argv[3])]
d.to_csv(sys.argv[2], index=False)
print(f'  {len(d)} stelle su {n0} sotto V<{sys.argv[3]}')
" "$CAT" "$OUT/cat_bright.csv" "$VMAX"

    run python3 bright_star_grid.py \
        --input "$OUT/cat_bright.csv" --outdir "$GRIDB" \
        --Tmin "$TMIN" --Tmax "$TMAX" --dt "$DT" \
        --lat-min "$LAT_MIN" --lat-max "$LAT_MAX" --lat-step "$LAT_STEP" \
        --workers "$WORKERS"

    say "0c. culture celesti di Stellarium"
    run python3 stellarium_skycultures.py --root "$SKY" --outdir "$SKYP"
fi

TRAJ=$GRID/trajectories.csv
FULL=$GRIDB/fullcat.csv

# =========================================================================
# 1. la Terra, senza stelle
# =========================================================================
if want 1; then
    say "1a. orientamento della Terra"
    run python3 earth_orientation.py \
        --Tmin "$TMIN" --Tmax "$TMAX" --dt "$DT" \
        --outdir "$OUT/orientation"

    say "1b. risultati nulli"
    run python3 null_results.py \
        --span "${TMIN#-}" --dt "$DT" --lat "$LAT" \
        --outdir "$OUT/nulls"

    say "1c. registro di rilevabilita'"
    run python3 detectability.py \
        --catalog "$CAT" --lat "$LAT" --pm-vmax "$VMAX" \
        --outdir "$OUT/detect"
fi

# =========================================================================
# 2. il cielo e l'osservatore
# =========================================================================
if want 2; then
    say "2a. buio utile per latitudine, epoca e magnitudine"
    run python3 night_visibility.py \
        --tmin "$TMIN" --tmax "$TMAX" --dt "$DT" \
        --lat "$LAT" --lat-min 0 --lat-max 75 --lat-step 1 \
        --mags 0 2 4 --trajectories "$TRAJ" --vmax "$VMAX" \
        --outdir "$OUT/night"

    say "2b. geometria del cielo"
    run python3 sky_geometry.py \
        --trajectories "$TRAJ" --lat "$LAT" \
        --lat-min "$LAT_MIN" --lat-max "$LAT_MAX" --lat-step 2 \
        --vmax 3.0 --outdir "$OUT/geometry"

    say "2c. orizzonte: Sole, Luna, Via Lattea, stelle"
    run python3 horizon_geometry.py \
        --trajectories "$TRAJ" --orientation "$OUT/orientation/orientation.csv" \
        --lat "$LAT" --vmax 3.0 --outdir "$OUT/horizon"
fi

# =========================================================================
# 3. i poli
# =========================================================================
if want 3; then
    say "3a. stelle polari"
    run python3 pole_stars.py \
        --trajectories "$TRAJ" --Tmin "$TMIN" --Tmax "$TMAX" \
        --vmax "$VMAX" --frontier-epochs 0 -5 -13 -26 -50 -100 \
        --outdir "$OUT/pole"

    say "3b. catene di stelle che puntano al polo"
    run python3 pointer_pairs.py \
        --trajectories "$TRAJ" --epoch-step 1.0 --vmax 3.0 \
        --gap-max 10.0 --max-stars 5 --collinear-tol 2.0 \
        --outdir "$OUT/pointers"
fi

# =========================================================================
# 4. costellazioni e asterismi
# =========================================================================
if want 4; then
    say "4a. conteggio di visibilita' per cultura"
    run python3 constellation_visibility.py \
        --trajectories "$TRAJ" --skycultures "$SKYP" \
        --epoch-step 1.0 --outdir "$OUT/visibility"

    say "4b. epoca dei canoni"
    run python3 canon_epoch.py \
        --trajectories "$TRAJ" --skycultures "$SKYP" \
        --epoch-step 1.0 --outdir "$OUT/canon"

    say "4c. deformazione delle costellazioni IAU"
    run python3 constellation_drift.py \
        --trajectories "$TRAJ" --skycultures "$SKYP" \
        --culture "$CULTURE" --min-stars 2 --ref-kyr 0 \
        --outdir "$OUT/drift"

    say "4d. vita degli asterismi, contro cielo casuale"
    run python3 asterisms.py \
        --trajectories "$TRAJ" --ref-epoch 0 --vmax "$VMAX" \
        --n-random 400 --outdir "$OUT/asterisms"
fi

# =========================================================================
# 5. il ciclo dell'anno, dal fullcat
# =========================================================================
if want 5; then
    say "5a. fascia lunare"
    run python3 lunar_band.py \
        --trajectories "$TRAJ" --vmax 3.0 --outdir "$OUT/moon"

    say "5b. ciclo annuale a $LAT"
    run python3 annual_cycle.py \
        --fullcat "$FULL" --lat "$LAT" --vmax 3.0 \
        --outdir "$OUT/annual"

    say "5c. calendario stellare a $LAT"
    run python3 stellar_calendar.py \
        --input "$FULL" --lat "$LAT" \
        --Tmin "$TMIN" --Tmax "$TMAX" --dt 1.0 --vmax "$VMAX" \
        --outdir "$OUT/calendar"

    say "5d. tabelle LaTeX del calendario"
    run python3 stellar_calendar_latex.py
fi

say "fatto. tutto sotto $OUT/"
