#!/bin/bash
# compile.sh
# Compila main.tex con lualatex, gestendo bibliografia, indice e riferimenti incrociati.
# La bibliografia usa \nocite{*} nel preambolo: tutte le voci di references.bib
# vengono stampate nella sezione finale "Fonti", anche se il testo non contiene
# richiami puntuali del tipo \citep/\citet (il libro è divulgativo e richiama
# le fonti in modo discorsivo). Servono comunque i quattro passaggi sotto perché
# bibtex legge l'.aux prodotto dal primo lualatex, e i successivi lualatex
# incorporano il .bbl e stabilizzano indice e riferimenti incrociati.
# Uso: ./compile.sh

set -e

MAIN="main"

echo "== Passaggio 1/4: lualatex =="
lualatex -interaction=nonstopmode -halt-on-error "${MAIN}.tex"

echo "== Passaggio 2/4: bibtex =="
bibtex "${MAIN}" || echo "Attenzione: bibtex ha restituito un avviso, controllare ${MAIN}.blg"

echo "== Passaggio 3/4: lualatex (aggiorna citazioni) =="
lualatex -interaction=nonstopmode -halt-on-error "${MAIN}.tex"

echo "== Passaggio 4/4: lualatex (aggiorna indice e riferimenti incrociati) =="
lualatex -interaction=nonstopmode -halt-on-error "${MAIN}.tex"

echo ""
echo "Compilazione completata: ${MAIN}.pdf"

# Pulizia file ausiliari (mantiene .tex, .bib e .pdf)
rm -f "${MAIN}".aux "${MAIN}".bbl "${MAIN}".blg "${MAIN}".log \
      "${MAIN}".out "${MAIN}".toc "${MAIN}".lof "${MAIN}".lot

echo "File ausiliari rimossi. Restano main.tex, references.bib, main.pdf e i sorgenti in chapters/."
