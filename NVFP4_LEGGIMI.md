# K-NVFP4 + V-NVFP4 per Strata

Implementazione sperimentale, 6 ottobre 2026. Base: lo ZIP originale
`Strata-source-20261006-140057(1).zip`, senza il precedente esperimento TQ4-V.

**Stato aggiornato:** CUDA 12.8 / RTX 5070 SM120, inferenza Q3 fino a circa
32k token e ottimizzazioni del kernel sono stati verificati in copie isolate.
L'integrazione e la validazione del main sono state completate su questo host. HIP e input completi
128k non sono stati verificati. Risultati e limiti aggiornati:
[docs/NVFP4_OPTIMIZATION.md](docs/NVFP4_OPTIMIZATION.md).

## Attivazione

Nel comando del motore compilato da questi sorgenti, sostituire le opzioni KV con:

```bash
--kv nvfp4 --kv-resident 0
```

Il nome `nvfp4` seleziona **entrambe** K e V con rotazione Hadamard randomizzata
a 256 dimensioni e formato NVFP4. Non serve un'opzione TurboQuant aggiuntiva.
I formati precedenti e le impostazioni predefinite restano disponibili.
Il drafter MTP rimane INT8. Lo streaming della KV in RAM non e' supportato per
questo nuovo formato. L'indexer e Gated DeltaNet mantengono i formati precedenti.

Il launcher accetta `--kv nvfp4` insieme a `--build`: i binari di release
preesistenti non contengono questa modifica. La configurazione del server deve
puntare al nuovo eseguibile. Non basta sostituire i file Python.

## Compilazione e controlli sulla 5070

Dalla cartella dei sorgenti, con CUDA Toolkit compatibile con SM120 e le dipendenze
di compilazione che usi gia' per Strata:

```bash
cmake -S . -B build-nvfp4 \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DSTRATA_TEST_NVFP4=ON
cmake --build build-nvfp4 --target strata kv_nvfp4_gpu_test kv_nvfp4_native_conversion --parallel
bash tools/run_nvfp4_gpu_checks.sh build-nvfp4
```

Conservare le opzioni GGML/native del proprio build funzionante: i comandi sopra
non sostituiscono eventuali percorsi locali, toolchain o librerie gia' necessari.
Per la sola suite kernel, senza il motore e senza scaricare GGML, vedere
`docs/NVFP4_KV.md`. Su Windows i due test sono normalmente in `build-nvfp4/Release`;
lo script Bash richiede un ambiente Bash. Nel manuale ci sono i comandi diretti.

Lo script esegue i due test e poi `memcheck`, `racecheck`, `initcheck` e
`synccheck`. Un test saltato, una GPU assente o un tool mancante interrompono lo
script: non vengono contati come un risultato positivo. Dopo questi controlli
rimangono da verificare qualita' e prestazioni con i tuoi pesi e contesti reali.

## Memoria calcolata

A 124.000 token, solo il payload K/V dei 12 layer QSA:

| Formato | GiB |
|---|---:|
| INT8 del sorgente originale | 1,4634 |
| Nuovo K-NVFP4 + V-NVFP4 | 0,8204 |

Risparmio: **658,45 MiB**, pari al **43,94%**. Questo non e' il consumo totale del
modello: indexer, GDN, MTP, padding delle pagine e buffer sono separati. Il layout
occupa 148 byte per testa, inclusa la scala FP32; non 144 byte.

## Cosa e' questa variante

E' una combinazione **RHT256 + NVFP4 ispirata a TurboQuant**. Non implementa
l'intero algoritmo del paper TurboQuant, il suo codebook Lloyd-Max o QJL.
Memorizza K/V in NVFP4; il prefill usa MMA FP16 con codici rappresentabili
esattamente e scale FP32. Non implementa QK/PV con MMA FP4 native.
La conversione FP4 nel kernel di append usa l'intrinseca CUDA quando disponibile.

Dettagli: `docs/NVFP4_KV.md`. Risultati e limiti: `NVFP4_VALIDATION.md`.

## Applicare la modifica

La patch non elimina gli asset della propria copia. Lo ZIP aggiornato non include
i due font WOFF2: conservare la propria cartella `serve/web/fonts` quando si
sostituiscono i sorgenti. `SNAPSHOT_INFO.json` descrive la base originale;
`NVFP4_MANIFEST.json` descrive questa modifica.
