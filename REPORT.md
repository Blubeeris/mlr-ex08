# Task 4 — lepší klasifikátor listov

## 1. Zadanie, ktoré som dala agentke

Pri zlepsovani klasifikatora som chcela zachovat povodny jednoduchy a prehladny kod a vychadzat z modelu, ktory sme vytvorili v Task 2. Agentke som zadala, aby skusala jednotlive vylepsenia postupne a vzdy menila iba jednu vec, aby sa dalo zistit, co malo na vysledok skutocny vplyv. Dolezite bolo tiez zachovat povodne subory a vysledky a porovnavat vsetky varianty s rovnakym baseline modelom. Chcela som, aby sa experimenty opakovali s viacerymi seedmi a aby sa vysledky kontrolovali nielen na nasich validacnych fotografiach, ale aj na upravenych a historickych fotografiach. Pri vyhodnocovani som chcela vychadzat iba zo skutocne nameranych vysledkov.

## 2. Protokol

| položka | hodnota |
|---|---|
| baseline | model z Task 2, presne ten, čo beží na stránke |
| vlastné dáta | `task4/data/leaves` — **244** tréningových, **62** validovacích fotiek (apple 17 / cherry 19 / chestnut 11 / maple 15 vo val) |
| historické dáta | `task4/data/hist/leaves` — **700** tréningových, **120** validovacích fotiek, iní ľudia, iné telefóny |
| poradie tried | `apple, cherry, chestnut, maple` (zhodné s `model_info.json`) |
| sieť | `mobilenet_v3_small(weights="DEFAULT")`, `classifier[3]` nahradené `Linear(in, 4)`; **1 521 956** trénovaných parametrov |
| tréning | `ImageFolder`, batch 32 so `shuffle`, Adam `lr=1e-3`, CrossEntropy bez label smoothingu, **15 epoch**, `torch.manual_seed(seed)` |
| seedy | 0, 1, 2 pred inicializáciou siete (konvencia zo zápisníka) |
| váhy v tabuľke | **posledná epoha** — rovnako ako vznikol model z Task 2 |
| množina `val` | 62 validovacích fotiek našej skupiny, bez augmentácie |
| množina `camera` | tie isté fotky, raz náhodne orezané / otočené / presfarbené so **seedom 12345**, vygenerované raz a zdieľané všetkými variantmi |
| množina `hist` | 120 cudzích fotiek (iní ľudia, iné telefóny) |
| šum | 1 fotka z 62 = **1.6 b.**; baseline sa sám medzi seedmi pohybuje o **14.5 b.** (val 0.710 – 0.855) |
| kritérium signálu | rozdiel počítam za merateľný len ak sa **rozpätie troch seedov** variantu neprekrýva s rozpätím baseline |
| prostredie | Python, `torch 2.15.0+cpu`, `torchvision 0.29.2`, `onnx 1.23.2`, `onnxruntime 1.31.1`, CPU (zápisník Task 2 bežal v Colabe na GPU) |

**Baseline nameraný priamo** (ty isté množiny, priamo s `.onnx` zo stránky):
`val 0.855` · `camera 0.742` · `hist 0.625`. Hodnota 0.855 je presne to, čo
hlásil zápisník na 15. epoche — teda moje množiny a preprocessing sú totožné
so zápisníkom. Zároveň to znamená, že nasadený model je lepší koniec
rozdelenia, nie priemer (priemer mojich 3 baseline behov je 0.785).

Takyto protokol som si vybrala preto, aby bolo mozne jednotlive zmeny ferovo porovnat s povodnym modelom z Task 2. Pouzitie troch seedov mi zaroven umoznilo zistit, ci je zlepsenie stabilne, alebo moze byt sposobene iba nahodnou inicializaciou modelu.

## 3. Výsledky

### Tabuľka 1 — posledná epoha (hlavné meranie)

Priemer ± smerodajná odchýlka a min–max medzi tromi seedmi. `hist bez
duplikátov` = historické fotky bez 4 obsahovo identických snímok (kap. 5).

| variant | seeds | val 62 | camera 62 | hist 120 | hist bez 4 duplikátov | Δ val | Δ camera |
|---|---|---|---|---|---|---|---|
| baseline | 3 | 0.785 ±0.059 (0.710–0.855) | 0.737 ±0.033 (0.694–0.774) | 0.569 ±0.044 (0.508–0.608) | 0.563 ±0.045 (0.500–0.603) | — | — |
| **Task 2 nasadený model** (1 meranie) | 1 | 0.855 | 0.742 | 0.625 | — | +0.070 | +0.005 |
| geom | 3 | 0.796 ±0.073 (0.710–0.887) | 0.785 ±0.062 (0.726–0.871) | 0.525 ±0.051 (0.458–0.583) | — | +0.011 | +0.048 |
| color | 3 | 0.796 ±0.027 (0.758–0.823) | 0.715 ±0.040 (0.661–0.758) | 0.586 ±0.008 (0.575–0.592) | — | +0.011 | −0.022 |
| both_aug | 3 | 0.774 ±0.013 (0.758–0.790) | 0.839 ±0.046 (0.774–0.871) | 0.517 ±0.007 (0.508–0.525) | — | −0.011 | +0.102 |
| freeze_head | 3 | 0.785 ±0.015 (0.774–0.806) | 0.747 ±0.038 (0.694–0.774) | 0.442 ±0.018 (0.417–0.458) | — | +0.000 | +0.011 |
| epochs30 | 3 | 0.801 ±0.033 (0.758–0.839) | 0.699 ±0.068 (0.629–0.790) | 0.542 ±0.027 (0.508–0.575) | — | +0.016 | −0.038 |
| cosine30 | 3 | 0.849 ±0.030 (0.806–0.871) | 0.785 ±0.030 (0.742–0.806) | 0.536 ±0.010 (0.525–0.550) | — | +0.065 | +0.048 |
| label_smooth | 3 | 0.812 ±0.040 (0.758–0.855) | 0.796 ±0.008 (0.790–0.806) | 0.578 ±0.017 (0.558–0.600) | — | +0.027 | +0.059 |
| dropout40 | 3 | 0.737 ±0.033 (0.694–0.774) | 0.715 ±0.040 (0.661–0.758) | 0.536 ±0.031 (0.492–0.558) | — | −0.048 | −0.022 |
| adamw_wd | 3 | 0.801 ±0.015 (0.790–0.823) | 0.731 ±0.050 (0.661–0.774) | 0.519 ±0.027 (0.483–0.550) | — | +0.016 | −0.005 |
| hist_added | 3 | 0.866 ±0.042 (0.806–0.903) | 0.753 ±0.059 (0.677–0.823) | 0.897 ±0.014 (0.883–0.917) | 0.894 ±0.015 (0.879–0.914) | +0.081 | +0.016 |
| hist_aug | 3 | 0.774 ±0.105 (0.629–0.871) | 0.785 ±0.088 (0.661–0.855) | 0.831 ±0.087 (0.708–0.900) | — | −0.011 | +0.048 |
| scratch | 3 | 0.720 ±0.008 (0.710–0.726) | 0.403 ±0.026 (0.371–0.435) | 0.392 ±0.024 (0.358–0.408) | — | −0.065 | −0.333 |
| scratch_bn_aug | 3 | 0.704 ±0.008 (0.694–0.710) | 0.661 ±0.035 (0.613–0.694) | 0.492 ±0.025 (0.458–0.517) | — | −0.081 | −0.075 |
| scratch_gap | 3 | 0.629 ±0.046 (0.565–0.661) | 0.586 ±0.038 (0.532–0.613) | 0.383 ±0.025 (0.350–0.408) | — | −0.156 | −0.151 |
| combined | 3 | 0.849 ±0.027 (0.823–0.887) | 0.892 ±0.027 (0.855–0.919) | 0.583 ±0.053 (0.542–0.658) | 0.580 ±0.053 (0.543–0.655) | +0.065 | +0.156 |
| combined_hist | 3 | 0.849 ±0.030 (0.806–0.871) | 0.876 ±0.038 (0.823–0.903) | 0.894 ±0.004 (0.892–0.900) | 0.891 ±0.004 (0.888–0.897) | +0.065 | +0.140 |
| combined_ac | 3 | 0.828 ±0.027 (0.790–0.855) | 0.909 ±0.020 (0.887–0.935) | 0.603 ±0.004 (0.600–0.608) | 0.606 ±0.004 (0.603–0.612) | +0.043 | +0.172 |
| combined_ac_hist | 3 | 0.806 ±0.060 (0.742–0.887) | 0.866 ±0.020 (0.839–0.887) | 0.908 ±0.018 (0.883–0.925) | 0.905 ±0.019 (0.879–0.922) | +0.022 | +0.129 |

### Tabuľka 2 — váhy najlepšej epochy (separátne, nie je porovnateľné s tabuľkou 1)

Epocha sa vyberala podľa `val`, takže `val_best` je optimistické číslo. Camera a
hist sú s tými istými váhami zmerané preto, aby bolo vidno, o koľko sa výber
epochy pomýli.

| variant | epochs | najlepšia epoha (seed0 / seed1 / seed2) | val_best | camera_best | hist_best | Δ camera oproti poslednej epoche |
|---|---|---|---|---|---|---|
| baseline | 15 | 11 / 14 / 4 | 0.839 ±0.026 (0.806–0.871) | 0.758 ±0.013 (0.742–0.774) | 0.583 ±0.012 (0.575–0.600) | +0.022 |
| hist_added | 15 | 15 / 13 / 9 | 0.892 ±0.015 (0.871–0.903) | 0.731 ±0.027 (0.694–0.758) | 0.908 ±0.014 (0.892–0.925) | −0.022 |
| combined | 30 | 17 / 16 / 1 | 0.860 ±0.030 (0.839–0.903) | 0.866 ±0.088 (0.742–0.935) | 0.569 ±0.027 (0.533–0.600) | −0.027 |
| combined_hist | 30 | 19 / 16 / 21 | 0.903 ±0.035 (0.855–0.935) | 0.855 ±0.013 (0.839–0.871) | 0.914 ±0.008 (0.908–0.925) | −0.022 |
| combined_ac | 30 | 18 / 18 / 1 | 0.866 ±0.027 (0.839–0.903) | 0.849 ±0.076 (0.742–0.903) | 0.611 ±0.048 (0.550–0.667) | −0.059 |
| combined_ac_hist | 30 | 7 / 17 / 23 | 0.892 ±0.030 (0.871–0.935) | 0.833 ±0.033 (0.790–0.871) | 0.900 ±0.012 (0.892–0.917) | −0.032 |

### Čas a veľkosť

| variant | čas behu (s, priemer 3 seedov) | trénovaných parametrov | tréningových fotiek |
|---|---|---|---|
| baseline | 20.2 | 1 521 956 | 244 |
| freeze_head | 6.6 | 4 100 | 244 |
| scratch | 10.0 | 208 292 | 244 |
| scratch_bn_aug | 19.0 | 208 644 | 244 |
| scratch_gap | 18.9 | 61 124 | 244 |
| cosine30 / epochs30 | 35.9 / 36.4 | 1 521 956 | 244 |
| combined / combined_ac | 48.6 / 47.4 | 1 521 956 | 244 |
| hist_added / hist_aug | 69.7 / 92.4 | 1 521 956 | 944 |
| combined_hist / combined_ac_hist | 178.9 / 172.7 | 1 521 956 | 944 |

## 4. Čo ktorá zmena je a čo namerila

| variant | čo presne je zmenené oproti baseline | Δ camera | Δ val | Δ hist | záver |
|---|---|---|---|---|---|
| geom | tréningové fotky navyše `RandomResizedCrop(0.6–1.0)`, `HorizontalFlip`, `VerticalFlip`, `Rotation(20)` | +0.048 | +0.011 | −0.044 | Geometrická augmentácia mierne zlepšila výsledky na camera množine, ale na historických fotografiách sa výsledok zhoršil.|
| color | tréningové fotky navyše `ColorJitter(0.3, 0.3, 0.2, 0.05)` | −0.022 | +0.011 | +0.017 | Farebná augmentácia nepriniesla výrazné zlepšenie na camera množine, ale mierne pomohla na historických fotografiách.|
| both_aug | `geom` + `color` spolu | +0.102 | −0.011 | −0.052 |Kombinácia geometrickej a farebnej augmentácie výrazne zlepšila výsledky na camera množine, ale nepomohla na validačnej ani historickej množine. |
| freeze_head | zmrazené všetky vrstvy okrem `classifier` (4 100 parametrov) | +0.011 | +0.000 | −0.127 |Zmrazenie väčšiny vrstiev neprinieslo výrazné zlepšenie a výsledky na historických fotografiách boli podstatne horšie. |
| epochs30 | 30 epoch namiesto 15, bez scheduleru | −0.038 | +0.016 | −0.027 |Samotné zvýšenie počtu epoch na 30 neprinieslo jednoznačné zlepšenie. |
| cosine30 | 30 epoch + `CosineAnnealingLR` | +0.048 | +0.065 | −0.033 |Cosine scheduler spolu s 30 epochami zlepšil výsledky na validačnej aj camera množine. |
| label_smooth | `CrossEntropyLoss(label_smoothing=0.1)` | +0.059 | +0.027 | +0.009 | Label smoothing priniesol stabilné zlepšenie na camera množine a mierne zlepšenie aj na historických fotografiách.|
| dropout40 | `classifier[2]` dropout nastavený na 0.4 | −0.022 | −0.048 | −0.033 |Zvýšenie dropout na 0.4 výsledky skôr zhoršilo, pretože presnosť klesla na všetkých troch množinách. |
| adamw_wd | `AdamW(lr=1e-3, weight_decay=0.05)` namiesto Adam | −0.005 | +0.016 | −0.050 | Použitie AdamW s weight decay neprinieslo oproti pôvodnému Adam jednoznačné zlepšenie.|
| hist_added | do tréningu pridaných 700 historických fotiek (944 celkom) | +0.016 | +0.081 | +0.328 |Pridanie historických fotografií výrazne zlepšilo výsledky na historickej množine a zároveň pomohlo aj na našej validácii. |
| hist_aug | `hist_added` + `both_aug` | +0.048 | −0.011 | +0.262 | Kombinácia historických dát a augmentácie zlepšila výsledky na historických a camera fotografiách, ale na validácii nepriniesla zlepšenie.|
| scratch | vlastná sieť z Task 3 (LeafNet, 3× conv bez BN, flatten) | −0.333 | −0.065 | −0.177 |Vlastná sieť trénovaná od začiatku dosahovala výrazne horšie výsledky ako predtrénovaný MobileNet. |
| scratch_bn_aug | LeafNet s BatchNorm + `both_aug` | −0.075 | −0.081 | −0.077 | BatchNorm a augmentácia pomohli vlastnej sieti oproti základnému scratch variantu, ale stále nedosiahla výsledky MobileNetu.|
| scratch_gap | LeafNet s BN a `AdaptiveAvgPool2d` namiesto flatten | −0.151 | −0.156 | −0.186 |Použitie AdaptiveAvgPool2d namiesto flatten výsledky ešte zhoršilo. |
| combined | `both_aug` + 30 epoch + cosine + AdamW/0.05 | +0.156 | +0.065 | +0.014 |Kombinácia viacerých vylepšení výrazne zlepšila výsledky na camera množine a zlepšila aj validáciu. |
| combined_hist | `combined` + historické dáta | +0.140 | +0.065 | +0.325 |Pridanie historických dát ku kombinovanému modelu výrazne zlepšilo výsledky na historických fotografiách. |
| combined_ac | `both_aug` + 30 epoch + cosine (bez AdamW) | +0.172 | +0.043 | +0.034 |Tento variant dosiahol najlepšiu priemernú presnosť na camera množine zo všetkých testovaných variantov. |
| combined_ac_hist | `combined_ac` + historické dáta | +0.129 | +0.022 | +0.339 |Kombinácia augmentácie, cosine scheduleru a historických dát dosiahla veľmi dobré výsledky na historických fotografiách, ale na camera množine bola o niečo slabšia ako combined_ac. |

Z vysledkov som zistila, ze najviac pomohla kombinacia geometrickej a farebnej augmentacie, najma pri testovani na camera mnozine. Velky rozdiel prinieslo aj pridanie historickych fotografii, ktore vyrazne zlepsilo klasifikaciu na historickych datach. Naopak, vlastna siet trenovana od zaciatku nedosahovala take dobre vysledky ako predtrénovany MobileNet a samotne zvysenie dropout alebo zmrazenie vrstiev velmi nepomohlo. Prekvapilo ma hlavne to, ze model s najlepsimi vysledkami na historickych fotografiach nemusel byt zaroven najlepsi pri praci s fotografiami podobnymi zaberom z kamery.

## 5. Merané skutočnosti, ktoré obmedzujú závery

Všetky štyri body sú overené priamo (skript, ktorý to zmeral, je uvedený):

1. **`val` je malá množina.** 62 fotiek, šum 1.6 b. na fotku; baseline sa medzi
   seedmi pohybuje 0.710 – 0.855. Na `val` sa takmer každý rozdiel v tabuľke 1
   prekrýva s baseline. (`table.py`)
2. **4 duplikáty v historickej zbierke.** `hist/train` obsahuje 4 fotky, ktoré
   sú obsahovo identické so 4 fotkami v `hist/val`
   (`maple_004/027/028/029.jpg` = `maple_161/057/047/012.jpg`, SHA256 po
   preškálovaní na 32×32). Nadhodnocuje `hist` max o 4/120 = 3.3 b. Preto
   tabuľka 1 obsahuje stĺpec `hist bez 4 duplikátov`. V lastných dátach
   prieniky nie sú: group train ∩ group val = 0 fotiek, hist ∩ group = 0
   fotiek. (`audit.py`, `hist_dups.json`)
3. **Kontrolná fotka je označená validovaná fotka.** `check_image.jpg` je
   `val/apple/apple_000.jpg` (porovnanie obsahu všetkých 306 fotiek, najbližšia
   zhoda 0.29). Nové modely — aj čistý baseline bez jedinej zmeny — ju trafia len
   v 1 z 3 seedov: apple 97.2 % / cherry 99.8 % / cherry 93.4 %.
   `check_expected.json` je teda test integrity nasadenia, nie test správnosti.
   (`audit.py`, `final_compare.py`)
4. **`camera` nie je fotoaparát.** Je to jedna fixná augmentácia tej istých
   62 fotiek (seed 12345). Skutočné zábery z telefónu som nemerila.
   (`exp.py: camera_like`)

Pri vyhodnocovani vysledkov som si uvedomila, ze nie vsetky rozdiely medzi modelmi musia znamenat skutocne zlepsenie. Nasa validacna mnozina obsahuje iba 62 fotografii, takze aj jedna nespravna predikcia dokaze ovplyvnit vyslednu presnost. Preto som porovnavala aj vysledky z troch seedov a sledovala, ci sa ich rozpätia prekryvaju s baseline. Zistila som tiez, ze v historickych datach sa nachadzaju styri duplicitne fotografie, co moze mierne nadhodnocovat vysledky. Kontrolna fotografia navyse patri priamo do validacnej mnoziny, takze jej spravne rozpoznanie nepovazujem za nezavisly dokaz kvality modelu. Z tychto dovodov mozem povedat, ze niektore varianty dosiahli lepsie vysledky na konkretnych testovacich mnozinach, ale nemozem tvrdit, ze budu rovnako dobre fungovat na vsetkych skutocnych fotografiach.

## 6. Model, ktorý by išiel do telefónu

Kandidát: **`combined_ac_hist`, seed 0**, váhy poslednej epochy.

| čo | hodnota |
|---|---|
| PyTorch checkpoint | `rerun_combined_ac_hist_seed0.pt` |
| ONNX | `leaf_model.onnx`, 6.10 MB, opset 17 |
| vstup ONNX | `image`, `[1, 3, 96, 96]`, `float32` — sedí na to, čo stránka posiela (`session.inputNames[0]`, stredový štvorec → 96×96 → /255 → mean/std → CHW) |
| výstup ONNX | `logits`, `[1, 4]`, poradie tried `apple, cherry, chestnut, maple` |
| `model_info.json` | **nezmenené** (triedy, 96, mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`) — stránku netreba upravovať |
| zhoda PyTorch ↔ ONNX | **244/244 rovnakých predikcií** (62 val + 62 camera + 120 hist), najväčší rozdiel v logitoch 2.05e-05 (val), 4.77e-05 (camera), 2.86e-05 (hist) |
| presnosti z ONNX | val 0.742 · camera 0.871 · hist 0.917 (zhodné s tréningom) |
| kontrolná fotka | cherry 68.5 % / apple 31.5 % pri štítku apple → `check_expected.json` by sa musel pregenerovať |

**Stabilita na jedinom skutočnom zábere z telefónu** (z `mobil.png` som vystrihla
okienko kamery 1188×1188 px a 40-krát ho posunula o max. 40 px, teda 3 % záberu;
podiel snímok, kde model povie `apple`):

| model | 40 posunov |
|---|---|
| Task 2 nasadený | 58 % (priemer apple 49.7 %) |
| baseline seed0 / 1 / 2 | 0 % / 100 % / 28 % |
| combined seed0 / 1 / 2 | 100 % / 95 % / 95 % |
| combined_ac seed0 / 1 / 2 | 80 % / 98 % / 100 % |
| combined_hist seed0 / 1 / 2 | 0 % / 2 % / 2 % |
| combined_ac_hist seed0 / 1 / 2 | 15 % / 8 % / 10 % |
| hist_added seed0 / 1 / 2 | 0 % / 0 % / 8 % |

Druh listu na tom zábere nie je istý — modely trénované s historickými dátami sa
prikláňajú k `cherry`, modely len s vlastnými dátami k `apple`.

**Test v aplikácii na telefóne.** Novy model som priamo v aplikacii na telefone netestovala. Urobila som iba dodatocne meranie na jednom skutocnom zabere z telefonu, pri ktorom som sledovala stabilitu predikcii pri malych posunoch obrazu. Kedze vysledky neboli dostatocne na jednoznacne rozhodnutie o nasadeni a novy model nebol overeny priamo v aplikacii, na stranke som zatial ponechala povodny model z Task 2.

## 7. Podklady a reprodukcia

V adresári [`task4/`](task4/) v tomto repozitári je:

- `results.jsonl` (51 behov, 17 variantov) a `results2.jsonl` (18 behov:
  12 zopakovaných + 6 nových; so starými tvoria 57 riadkov = 19 variantov ×
  3 seedy) — jeden riadok = jeden beh so všetkými nastaveniami aj nameranými
  presnosťami,
- `table_v2.md` — obe tabuľky z kapitoly 3, presne v tej podobe, v akej sú vyššie,
- `exp.py` — tréning aj meranie; varianty sú jeden slovník, baseline je model
  z Task 2, `--save` ukladá váhy poslednej epochy,
- `table.py` — spriemeruje seedy a vyhodí markdown tabuľky.

```bash
cd task4
python -I exp.py --variants combined_ac_hist --seeds 0 1 2 --out results2.jsonl --save rerun_combined_ac_hist
python -I table.py results.jsonl results2.jsonl > table_v2.md
```

`exp.py` počíta s lokálnymi cestami k fotkám (`task4/data/leaves`,
`task4/data/hist/leaves`), ktoré v repozitári nie sú — surové fotky ostali
u nás. Kontroly, čo sú v kapitole 5, a export modelu robia skripty `audit.py`,
`repro_check.py`, `final_compare.py`, `package_model.py`, `leaf_robust.py`,
`crop_sensitivity.py`, `deploy_compare.py` a `seeds_check.py`; tie som do
repozitára nedávala, lebo pracujú s tými istými lokálnymi cestami. Ich výsledok
je ale v `results*.jsonl` a v kapitolách 3 a 5, teda vrátane počtu fotiek,
seedu a času behu.

Reprodukovateľnosť: 12 behov, ktoré bežili dvakrát v samostatných procesoch, sa
zhoduje na val aj camera aj hist na desatinné miesta.

Rozhodla som sa pridat do repozitara aj hlavne experimentalne skripty a vysledky, aby bolo mozne moje merania v buducnosti zopakovat a skontrolovat. Pridala som najma exp.py, table.py, table_v2.md a subory results.jsonl a results2.jsonl, vse v adresari task4/. Vdaka tomu bude jasne, ake zmeny som skusala, ako som ziskala vysledne tabulky a na zaklade coho som jednotlive modely porovnavala.
