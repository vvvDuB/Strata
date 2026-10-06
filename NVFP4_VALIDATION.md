> Historical validation of the supplied source archive, not the final installation report.
> For subsequent real-CUDA/Q3 checks and main integration, see [docs/NVFP4_OPTIMIZATION.md](docs/NVFP4_OPTIMIZATION.md).

# Strata RHT256-NVFP4: rapporto di validazione

Data: 6 ottobre 2026. Nuova opzione: `--kv nvfp4 --kv-resident 0`.

## Esito

**Implementazione sperimentale disponibile; validazione GPU e modello ancora
mancanti.** La nuova KV usa NVFP4 su K e V dei layer QSA principali, con RHT256.
Il drafter MTP resta INT8. Non e' una certificazione di assenza di regressioni,
qualita' invariata o accelerazione sulla RTX 5070.

Ambiente effettivamente usato: Linux, GCC 14.2, Clang 17, CMake 3.31.6.
`nvcc`, `hipcc`, `nvidia-smi`, device NVIDIA e pesi del modello non disponibili.
Non sono stati prodotti PTX, cubin o eseguibili CUDA/HIP. Le verifiche di sintassi
con dichiarazioni sostitutive non cambiano questa limitazione.

## Provenienza e contenuto

Base: `Strata-source-20261006-140057(1).zip`, SHA-256:

```text
b153c2588d435656a0fb9de97aba1cc8055cf328b7eaf45baeea68f929f5b4ca
```

Il motore parte da questa base, senza il precedente TQ4-V. Il confronto di
regressione usa il file `setup.py` originale estratto dalla stessa base.
Il file originale `SNAPSHOT_INFO.json` resta l'inventario della base; le aggiunte
e le modifiche NVFP4 sono elencate separatamente in `NVFP4_MANIFEST.json`.

La patch non elimina file originali. Lo ZIP aggiornato non include i due asset
`serve/web/fonts/*.woff2`: per conservare tutti gli asset, applicare la patch
alla propria copia o mantenere la cartella font originale. Non sono inclusi
binari compilati durante questo lavoro, cache di build o cartelle `.git`.

## Test realmente eseguiti

| Verifica | Risultato |
|---|---|
| Suite CMake host autonoma | 17/17 passati |
| Codec host con ASan e UBSan | 474.686 controlli passati, nessun errore segnalato |
| Snapshot/prefix con trasferimenti host simulati, ASan e UBSan | 9.900 controlli passati, nessun errore segnalato |
| Roundtrip del file conversazione, incluso formato 5 | 6.851 controlli passati |
| Comparatore dei logits, inclusa modalita' native raw-first | 17/17 passati |
| Test del launcher specifici NVFP4 | 5/5 passati |
| Build CMake CPU estesa | Compilazione riuscita; 32/34 test passati |
| Suite launcher preesistente, originale e modificata | 179 test ciascuna; stessi 47 fallimenti, zero errori |
| Configurazioni predefinite del launcher | 25 fixture identiche |
| Python bytecode, sintassi Bash e `git diff --check` | Passati |

I due fallimenti della suite CPU estesa sono `expert_parity` e `pool_test`:
richiedono `pack/full/experts.bin`, assente. I log riportano l'impossibilita' di
leggere il primo esperto. Non li conto come passati o come validazione del modello.

I 47 fallimenti della suite launcher sono riproducibili con il file originale;
sono stati confrontati gli identificatori esatti dei test, non soltanto il totale.
Questo non dimostra che qualsiasi configurazione possibile del launcher sia
priva di regressioni. Le 25 fixture golden e i cinque nuovi casi sono un controllo
separato sui percorsi effettivamente esercitati.

Il codec host confronta conversioni FP4/FP8 con un oracolo nearest-even,
midpoint e valori vicini, tutte le basi della rotazione, inverse/norme/prodotti
scalari, righe casuali e con outlier, zero, piccoli valori e non-finiti. Il valore
NMSE sintetico stampato dal test (0,0095099 aggregato pesato; massimo 0,0162970)
riguarda esclusivamente le fixture numeriche del codec, non il modello Qwen.

I test snapshot/prefix compilano il codice di produzione con un trasporto
`memcpy` isolato. Verificano formato, geometria, pagine parziali, ripristino
A/B/A, mismatch prima delle scritture e alcuni errori di trasferimento. Non
simulano la concorrenza CUDA, il driver o i kernel.

## Verifiche di sintassi supplementari

Clang ha analizzato le parti host/device delle tre unita' CUDA modificate e del
test di conversione, piu' la parte host del test GPU, con header contenenti solo
dichiarazioni sostitutive. I nove controlli terminano con codice zero.

**Non e' una compilazione CUDA reale:** nessun header SDK autentico, nessuna
verifica dell'ABI delle intrinseche o delle istruzioni PTX, nessun collegamento,
allocazione registri, esecuzione o test di sincronizzazione. La macro
`__CUDA_ARCH__` impostata in questi controlli serve solo a leggere il ramo di
preprocessore interessato. Non attesta supporto o generazione SM120.
Il riepilogo contiene tutti i comandi, per evitare di confondere questi controlli
con un build `nvcc`.

## Test GPU predisposti, NON eseguiti

`STRATA_TEST_NVFP4=ON` aggiunge due eseguibili senza bisogno dei pesi:

- `kv_nvfp4_native_conversion`: usa lo stesso helper della scrittura di produzione;
  verifica ordine dei nibble, midpoint positivi/negativi, saturazione, NaN/Inf,
  signed zero, subnormali e numerose coppie di valori. Su Blackwell un ramo nativo
  non esercitato viene segnalato, non considerato un successo.
- `kv_nvfp4_gpu_test`: append/rotazioni/gather reali, batch/step/decode e 22 fixture
  di attention prompt, riferimento CPU in doppia precisione sui valori NVFP4
  memorizzati, canary, pagine permutate e parziali, indici invalidi, selezioni fino
  a 2051 token, width dinamica e replay CUDA Graph con posizioni diverse.

`tools/run_nvfp4_gpu_checks.sh` richiede il successo dei due eseguibili e dei
quattro sanitizer device (`memcheck`, `racecheck`, `initcheck`, `synccheck`).
Un eseguibile assente, un tool assente o un test saltato fanno fallire il gate.
I test CPU non sostituiscono questo passaggio.

Rimangono inoltre da eseguire sull'engine completo: confronto INT8/NVFP4 a parita'
di pesi e token, resume delle conversazioni, accettazione/rifiuto speculativo,
qualita' su contesti lunghi, throughput e VRAM di picco. Le soglie del test kernel
misurano parita' con l'attention quantizzata scelta, non qualita' rispetto a INT8.

## Numeri di memoria e limiti prestazionali

Payload per testa: 128 byte codici E2M1 + 16 byte scale E4M3 + 4 byte scala FP32,
cioe' 148 byte. Due teste KV, due pool: 592 byte/token/layer principale.
A 124.000 token e 12 layer: 0,820398 GiB contro 1,463413 GiB INT8;
risparmio calcolato 658,447 MiB (43,9394%). Sono esclusi indexer, GDN, MTP,
page padding, allineamenti, pesi e buffer. Il contatore del launcher include
il draft INT8: 8160 byte per token per la componente stimata.

La KV e' NVFP4, ma le operazioni di attention non sono MMA FP4 native. Il nuovo
prefill usa codici E2M1 rappresentabili esattamente come operand FP16 e scale
FP32. Il decode legge le righe compresse e lavora in FP32. Una diversa
quantizzazione e un risparmio di banda possono comunque peggiorare il tempo
kernel; non e' stato misurato alcuno speedup.

## Riproduzione e attivazione

I comandi di build, gate GPU, controlli host, benchmark e confronto dei logits
sono in `docs/NVFP4_KV.md`. La guida rapida italiana e' `NVFP4_LEGGIMI.md`.
Per i pack nativi, usare `STRATA_DUMP_FIRST_LOGITS` e il comparatore con
`--raw-first`: il normale dump per-token non e' raggiunto da quel percorso
nell'engine originale. Un singolo vettore dopo il prompt e' una diagnostica,
non una misura di perplexity.

La modifica resta opt-in. Compilare i sorgenti, verificare i gate sul dispositivo
e puntare il server al nuovo binario prima di selezionare `--kv nvfp4`.
