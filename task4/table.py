"""Z results.jsonl zhrnie markdown tabuľku pre REPORT.md.

Riadiť sa dá podľa tohto výstupu, text reportu si musíš napísať sám (zadanie:
REPORT.md píše študent). Skript len spriemeruje seed-y a ukáže rozdiel oproti
základu — aby bolo vidno, či zmena je väčšia ako šum.

Dva súbory za sebou: staršie potreby vpravo ne prepíšu, takže
  python -I table.py results.jsonl results2.jsonl > table_v2.md
ukáže 51 pôvodných behov a prebehnuté riadky (s meraniami navyše) v nich
nahradia tie staré.

Hlavná tabuľka sú váhy POSLEDNEJ epochy — rovnako ako vznikol model z Task 2.
Tabuľka najlepšej epochy je oddelená, lebo epocha sa vyberala podľa val, teda
je to optimistické číslo a nesmie sa miešať s poslednou epochou.
"""

import json
import statistics
import sys
from collections import defaultdict

# Windows konzola (cp1250) nevie UTF-8, a -I ignoruje PYTHONIOENCODING -> nastavim prúd ručne
sys.stdout.reconfigure(encoding="utf-8")

cesty = sys.argv[1:] or ["results.jsonl"]
podla_seedu = {}
for cesta in cesty:
    for line in open(cesta, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        k = (r["variant"], r["seed"])
        podla_seedu[k] = r  # neskorší súbor vyhráva
rows = list(podla_seedu.values())

order, agg = [], defaultdict(list)
for r in rows:
    key = r["variant"]
    if key not in order:
        order.append(key)
    agg[key].append(r)


def mean_spread(vals):
    """Priemer a rozptyl medzi seedmi ako 'priemer ±smerodajka (min–max)'."""
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return (f"{statistics.mean(vals):.3f} ±{statistics.pstdev(vals):.3f} "
            f"({min(vals):.3f}–{max(vals):.3f})")


# ---- TABUĽKA 1: posledná epocha (hlavná, porovnateľná s Task 2) -------------
metriky = [("val_last", "val 62"), ("camera", "camera 62"), ("hist", "hist 120")]
if any("hist_excl" in r for r in rows):
    metriky.append(("hist_excl", "hist bez 4 duplikátov"))
base = agg.get("baseline")

print("## Tabuľka 1 — posledná epocha (hlavné meranie, rovnaký protokol ako Task 2)\n")
print("| variant | seeds | " + " | ".join(label for _, label in metriky) + " | Δ val | Δ camera |")
print("|---" * (len(metriky) + 4) + "|")
for key in order:
    rs = agg[key]
    cells = [key, str(len(rs))]
    deltas = {}
    for field, _ in metriky:
        # staršie behy (pred opravou) nemaju hist_excl - tam dam ciarku, nie priemer z nicoho
        vals = [r[field] for r in rs] if all(field in r for r in rs) else []
        cells.append(mean_spread(vals) if vals else "—")
        if base and field in rs[0] and field in base[0]:
            b = statistics.mean([r[field] for r in base])
            deltas[field] = statistics.mean(vals) - b
    for field in ["val_last", "camera"]:
        cells.append(f"{deltas[field]:+.3f}" if base and key != "baseline" else "—")
    print("| " + " | ".join(cells) + " |")

# ---- TABUĽKA 2: najlepšia epoha (separát, optimistické) ---------------------
ma_best = [k for k in order if "camera_best" in agg[k][0]]
if ma_best:
    print("\n## Tabuľka 2 — váhy najlepšej epochy (separátne, NIE porovnateľné s tabuľkou 1)\n")
    print("Epocha sa vyberala podľa val, takže val_best je optimistické číslo; camera aj hist")
    print("sú zmerané s tými istými váhami, aby sa ukázalo, o koľko sa líši výber epochy.\n")
    print("| variant | epochs | epoha najlepšia | val_best | camera_best | hist_best | Δ camera oproti poslednej |")
    print("|---|---|---|---|---|---|---|")
    for key in ma_best:
        rs = agg[key]
        delta = statistics.mean([r["camera_best"] - r["camera"] for r in rs])
        print(f"| {key} | {rs[0]['epochs']} | {sorted({r['best_epoch'] for r in rs})} "
              f"| {mean_spread([r['val_best'] for r in rs])} "
              f"| {mean_spread([r['camera_best'] for r in rs])} "
              f"| {mean_spread([r['hist_best'] for r in rs])} | {delta:+.3f} |")

print()
print("čas_behu_s (priemer):", {k: round(statistics.mean([r["seconds"] for r in agg[k]]), 1) for k in order})
print("pocet_trenovanych_parametrov:", {k: agg[k][0]["trainable"] for k in order})
print("pocet_trenovanych_fotiek:", {k: agg[k][0]["n_train"] for k in order})
