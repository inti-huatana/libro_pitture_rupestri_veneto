# I programmi di astronomia preistorica

Calcoli sul cielo dei 100 000 anni passati: dove stavano le stelle, che cosa
si vedeva e da dove, che cosa è cambiato abbastanza da lasciare traccia in una
tradizione e che cosa non è cambiato affatto.

Tutto discende da un unico catalogo Hipparcos con parallasse, moto proprio e
velocità radiale, propagato in tre dimensioni, e da un modello di precessione
a lungo termine (Vondrák et al. 2011, valido su ±200 kyr). Nessun programma
introduce costanti scelte a piacere: ogni soglia o è misurata dai dati, o è
una proprietà fisica citata, o è un parametro della riga di comando con il suo
valore giustificato nella docstring.

**Il catalogo di ingresso arriva a Hp = 6.5**, circa novemila stelle. È il
limite dell'occhio nudo in condizioni ideali, non quello del riconoscimento:
oltre la quarta magnitudine una stella non si distingue più dalle vicine, e
quasi tutti i programmi tagliano lì con `--vmax`. Il taglio è sempre esplicito
e sempre modificabile.

---

## Come si incastrano

```
   catalogo Hipparcos (Hp < 6.5)
            │
            ▼
   bright_star_grid.py ──► trajectories.csv   (posizioni, indipendenti dalla latitudine)
            │           └► fullcat.csv        (visibilità ed eventi eliaci, per latitudine)
            │
   ┌────────┴──────────────────────────────────────────────┐
   │                                                        │
   ▼ traiettorie                                            ▼ fullcat
   pole_stars, pointer_pairs, sky_geometry,                 annual_cycle,
   asterisms, lunar_band, night_visibility,                 stellar_calendar
   constellation_*, canon_epoch, horizon_geometry
            ▲
            │
   stellarium_skycultures.py ──► members.csv, constellations.csv, segments.csv

   earth_orientation.py ──► orientation.csv ──► horizon_geometry.py
   detectability.py, null_results.py: non leggono le traiettorie
```

---

## Generazione dei dati

### `bright_star_grid.py`
Il motore. Propaga ogni stella in coordinate rettilinee tridimensionali, la
precessa all'equatore di data, e per ogni latitudine della griglia calcola
classe di visibilità, levate e tramonti eliaci e acronici. Produce
`trajectories.csv` (una riga per stella ed epoca, senza latitudine),
`stars/HIP_*.csv` e la loro concatenazione `fullcat.csv` (una riga per stella,
epoca e latitudine).

Gli eventi d'orizzonte sono vettorizzati sulle epoche con bisezione al posto
di una radice scalare per stella; l'equivalenza con il motore precedente è
verificata da `validate_grid_engine.py` a 10⁻¹⁰ giorni su 26 880 confronti.

`fullcat.csv` è il file che esplode: righe = stelle × epoche × latitudini. Con
il catalogo completo va generato solo sul sottoinsieme riconoscibile a occhio.

### `stellarium_skycultures.py`
Legge l'albero delle culture celesti di Stellarium (`index.json`, non i vecchi
`.fab`) e ne estrae appartenenze e linee. Raccoglie i membri da **tre** fonti
distinte — le linee delle costellazioni, le àncore delle immagini e i nomi
propri `HIP nnnnn` — perché nessuna delle tre da sola è completa: con le sole
linee mancavano Arturo e Sirio. La colonna `source` dice da dove viene ogni
appartenenza.

---

## Il cielo e l'osservatore

### `night_visibility.py`
Quanto dell'anno il cielo è abbastanza buio per servire, per latitudine ed
epoca. Non usa i −18° convenzionali, che sono una soglia fotometrica, ma
l'**arcus visionis** già usato per le levate eliache, `AV = 10.5 + 1.4·V`: la
soglia diventa così una funzione della stella e non una scelta.

Il risultato che il criterio unico nascondeva: fra circa 50° e 56° di
latitudine esiste una stagione estiva in cui le stelle brillanti si vedono e i
membri deboli delle loro costellazioni no. Le figure si sfaldano.

L'anno è percorso in longitudine solare vera e pesato per il tempo, così il
totale annuo conserva il battimento di Milankovitch che una griglia uniforme
di giorni cancellerebbe.

### `detectability.py`
Il filtro che rende leggibile tutto il resto. Prende i tassi calcolati altrove
e li confronta con quattro soglie osservative — vernier 1′ (Mizar e Alcor
distano 11.8′), assoluta 10′ (pavimentata dalla rifrazione), disco solare 32′,
un giorno — e restituisce dopo quante generazioni ciascun fenomeno diventa
visibile. Produce CSV, due tabelle LaTeX `\input`-abili e due figure.

Due risultati capovolgono le assunzioni correnti: **l'equinozio è un marcatore
d'orizzonte dieci volte più preciso del solstizio**, perché al solstizio la
declinazione solare è stazionaria; e **la precessione è muta all'orizzonte e
clamorosa nel calendario**, il che spiega perché le tradizioni stellari si
datano e gli allineamenti litici quasi mai.

### `null_results.py`
L'altra colonna del registro: le quindici grandezze che non sono cambiate
abbastanza da essere notate, ciascuna con la sua variazione su ±100 kyr contro
la soglia che l'avrebbe registrata. Non legge le traiettorie.

Nutazione contro lunistizio, stesso periodo di 18.6 anni e ampiezze in
rapporto di migliaia. La Luna identica a se stessa. I periodi planetari
invarianti, perché giorno e anno si allungano insieme e il loro rapporto no.
Unica voce non nulla: il residuo metonico, riportato con la **scansione di
sensibilità** al tasso di dissipazione mareale, che non è costante.

---

## Punti di riferimento del cielo

### `pole_stars.py`
Qualità dell'orientamento ai due poli. La distanza polare **è** l'errore di
rilevamento, quindi non serve nessun punteggio composito: si tabulano la
migliore stella disponibile a ogni soglia di magnitudine, le successioni di
titolari con le loro durate, e la frontiera di Pareto fra vicinanza e
luminosità. La latitudine è deliberatamente esclusa: il polo non dipende da
dove si guarda.

### `pointer_pairs.py`
Catene di 2–5 stelle il cui allineamento, prolungato, trova il polo. Sfrutta
che ogni catena che punta al polo termina in una coppia che punta al polo:
si cercano le coppie terminali e si allungano all'indietro lungo il loro
cerchio massimo, evitando l'enumerazione combinatoria. Separazione massima fra
membri consecutivi configurabile (default 10°, perché su distanze maggiori la
retta si perde). Validazione: Merak→Dubhe a nord, Gacrux→Acrux a sud.

### `sky_geometry.py`
Cinque riferimenti che i programmi sul polo non toccano.

- **La stella dell'est.** Una stella a declinazione zero sorge esattamente a
  est *da qualunque latitudine*, mentre una stella polare va riletta a ogni
  grado percorso: per una popolazione in movimento vale di più. Verifica
  immediata: oggi è Mintaka, la stella occidentale della Cintura di Orione.
- **Il centro galattico.** Latitudine eclittica fissa a −5.6°, quindi la
  declinazione oscilla fra −29.0° e +17.8°. Oggi è al minimo. Il programma
  calcola le ore all'anno con il centro alto e il cielo buio, su griglia epoca
  × latitudine — perché la precessione porta anche la stagione della
  culminazione di mezzanotte in giro per il calendario.
- **Il polo dell'eclittica**, l'unico punto fermo, confrontato con l'attesa
  casuale √(4 ln2 / N) perché "non ha stelle vicino" da solo non significa
  niente con novemila stelle.
- **Il punto vernale** e i suoi vicini.
- **Lo squilibrio fra emisferi**: chi migra verso sud guadagna cielo.

### `earth_orientation.py`
Nessuna stella. Obliquità di data, polo galattico, azimut di levata, durata
delle stagioni al variare della longitudine del perielio. Produce
`orientation.csv`, che è l'ingresso di `horizon_geometry.py`.

### `horizon_geometry.py`
Dove Sole, Luna, Via Lattea e stelle incontrano l'orizzonte a una latitudine
data. Stelle che condividono il punto di levata con il Sole solstiziale o con
la Luna ai lunistizi, deriva del solstizio, inclinazione della Via Lattea.

---

## Costellazioni e asterismi

### `constellation_visibility.py`
Conteggio puro: quante stelle di una cultura erano visibili e quante no, in
funzione di latitudine ed epoca. Nessuno stimatore, nessun modello. È il punto
di partenza da cui deriva `canon_epoch.py`.

### `canon_epoch.py`
Quando può essere stato messo insieme il canone di ciascuna cultura, dato dove
viveva. Criterio fisso a 5° sull'orizzonte. Tiene **due conteggi mai sommati**:
le stelle nominate che stavano sotto il criterio (vincolo duro) e le stelle
brillanti visibili ma non nominate (vincolo debole). Produce la lista delle
stelle mai utilizzabili con altezza massima e latitudine eclittica, la larghezza
della finestra come numero di testa, e le mappe in tempo e latitudine.

Riscontro storico: la finestra greco-tolemaica esce fra −0.8 e +0.3 kyr,
delimitata da Acamar e Achernar — ed è noto che l'Eridano di Tolomeo finisce
ad Acamar perché Achernar da Alessandria non si vedeva.

### `constellation_drift.py`
Deformazione delle 88 costellazioni IAU sotto il moto proprio. Tre numeri dai
rapporti fra separazioni: scala, cambiamento totale, cambiamento di forma.
Nomi in latino, tavole per costellazione a passi di 10 kyr.

### `asterisms.py`
Le figure che qualcuno ha davvero indicato — Carro, Cintura, Triangolo
Estivo, Croce, Teiera — con i membri elencati per HIP. Misura dopo quanto la
deformazione diventa visibile, tolte rotazione e scala via Procrustes senza
riflessione: resta la stella che non sta più dove la mette il racconto.

Due ipotesi nulle: **gruppi casuali** della stessa taglia e numerosità, da cui
esce la vita naturale di un asterismo di caso; e **moti prestati**, membri veri
con il moto proprio di altre stelle, che isola se la longevità venga dal
viaggiare insieme. Verifica attesa: cinque delle sette stelle del Carro sono
del gruppo mobile dell'Orsa, Dubhe e Alkaid no, quindi il Carro deve rompersi
alle due estremità.

Il programma stampa la lista dei membri risolti con sigla di Bayer e
magnitudine: un HIP sbagliato salta all'occhio come stella della costellazione
sbagliata.

---

## Il ciclo dell'anno

### `annual_cycle.py`
Stelle imperiture (che non tramontano mai), durata dell'invisibilità fra
tramonto e levata eliaca, deriva del calendario. Legge `fullcat.csv`.

### `lunar_band.py`
Quali stelle brillanti la Luna poteva occultare, epoca per epoca. La fascia è
5.145° di inclinazione più parallasse e semidiametro, circa 6.4° di latitudine
eclittica. Non serve un'effemeride: non si chiede dove fosse la Luna una data
notte, che è inconoscibile, ma quale parte di cielo poteva raggiungere.

### `stellar_calendar.py` e `stellar_calendar_latex.py`
Calendario stellare stagionale a una latitudine, e le tabelle LaTeX
dell'appendice.

---

## Vicoli ciechi, tenuti solo come documentazione

Non entrano nella pipeline e non vanno eseguiti.

| file | perché |
|---|---|
| `constellation_age.py` | Datazione dalla zona vuota attorno al polo (metodo Ovenden–Roy). Lo stimatore massimizza una calotta vuota su tutti i centri possibili e finisce sul bordo dei dati; sostituito dal conteggio di `constellation_visibility.py` e `canon_epoch.py`. |
| `bright_star_longterm.py` | Motore a latitudine singola, superato da `bright_star_grid.py` (4.6× più veloce, risultati identici). Serve ancora come riferimento a `validate_grid_engine.py`. |
| `pole_stars_grid.py` | Elenca le N stelle più vicine ai poli. Una lista di vicinanze non dice quanto bene si potesse trovare il nord; sostituito da `pole_stars.py`. |

`validate_grid_engine.py` non è un vicolo cieco ma nemmeno parte della
pipeline: confronta i due motori e va eseguito solo quando `bright_star_grid.py`
viene modificato.

---

## Esecuzione

`./run_all.sh` esegue tutto in ordine. Le variabili in testa allo script
fissano catalogo, latitudine, intervallo temporale e passo.
