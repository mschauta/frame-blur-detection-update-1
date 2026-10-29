# R4S-1: ConvNeXt-Small és ConvNeXt-Base azonos recepttel

## Protokoll (rögzítve 2026-10-28, mielőtt bármely összehasonlító eredmény megvolt)

Előre ismert csak a tanítás közbeni validáció (epoch 29, kiválasztott frame-ek): AUC 0,9612 (Small) és 0,9621 (Base).

**Kérdés** (felhasználó, 2026-10-28): elég nagy-e a Small, hogy megtanulja, amit a Base; ha igen, mennyivel gyorsabb;
esetleg jobb. „Majd az összehasonlító tesztek döntik el.”

**Modellek:** a publikált fájlok és a publikált betöltő (frame-blur-detection-update-1, `62734f4`).
- Base: `round3_base_p99_ep029.pt` (release v1.0, sha `9840160f…`).
- Small: `round3_small_p99_ep029.pt` (release v1.1, sha `b715a163…`).
- Mindkettő az utolsó epoch (29), előre rögzítve (felhasználó: „Itt is az epoch29 legyen választva”).

**Nem tiszta gerinc-összevetés:** a Small maszkjavítással tanult (`mask: leak-free`). Ez csak maszkolt mintánál
számít; az alábbi tesztekben nincs maszk, így a két modell bemenete bitre azonos.

**Pontozás:** mindkét modell ugyanazzal a szkripttel (`tools/r4s1_small_base.py`), ugyanazon a gépen, GPU-n, a ConvNeXt
törzs bfloat16 autocast alatt, a fingerprint float32, batch 1 (a fotólétrán batch 8, mint a Base eredeti létráján).
Egy frame-et egyszer töltünk be, és mindkét modell ugyanazt a tömböt kapja.

### 1. Kimaradt filmek, VLM-címkék

- Anyag: a hat kimaradt film (3 validációs, 3 teszt) mind a 114 687 címkés frame-je. Éles = `sss` (plusz `ssb` az egy
  ssb-domináns filmben), blur = `bbb` + `bsb`, ahogy az Update 1 `heldout_films.json`-ja.
- Mérték: AUC filmenként, csoportonként (validációs, teszt, mind a hat); 95%-os intervallum klaszter-bootstrappal a
  snittekre (2000 újramintavétel, snittek visszatevéssel filmen belül, seed 20261028). Ugyanazok az újramintavételek
  mindkét modellre, így a Small − Base különbség páros.
- Működési pontok: mindkét modell a saját tárolt küszöbeivel (r90 / r95 / r98, a saját validációs részhalmazából):
  megtartott éles és elkapott blur arány csoportonként.
- A két modell egyezése: döntési egyezés r95-nél, Cohen-féle κ, a logitok Spearman-korrelációja.
- **Kontroll:** a Base újrapontozásának filmenkénti AUC-je egyezzen a publikált értékekkel (a korpuszpontozásból).
  Eltérésnél jelentjük, és a páros összevetés az újrapontozott Base-szel történik.

### 2. A mérés mint tanú

- A kép-logit Spearman-korrelációja a mért régiómozgással (`region_px`, v9 frame-tábla) azokon a címkés frame-eken,
  amelyeken van mérés; mindkét modell, páros bootstrap különbség (ugyanaz az eljárás, mint az 1.-ben).
- Másodlagos kiolvasás: M a cellatérképből (a belső cellák relu-átlaga), mert a v9-nél a Base 0,691-es referenciája
  így készült.

### 3. Fotólétra

- A 300 nem látott etalonfotó, fotónként egy rögzített irány, L = 0 … 30 px egyenletes kameramozgás, H.264 CRF 23
  (`tools/probe_round3.py`: ugyanazok a fotók, irányok, hosszak). Mindkét modell ugyanazokat a képeket kapja.
- Mérték: a blurnak mondott arány L szerint a saját küszöbökön, az 50%-os átfordulás helye, a medián logit, a
  pár-sorrend 0–3 és 3–16 px között, a szigorú növekedés 0–1–2–3 px-en.
- **Kontroll:** a Base egyezzen a publikált `photo_ladder_epoch29.json`-nal.

### 4. Sebesség

- Ugyanazon a gépen (RTX 5090), egy 1080p frame-mel, a `classify.py` útján: GPU, bfloat16 autocast; batch 1 és 8.
- 30 bemelegítő kör után 200 mért kör, CUDA-szinkronizálással; kép/s, ms/kép, csúcsmemória.
- Csak a modell (fingerprint, törzs, pooling); a képbetöltés és a dekódolás nincs benne. A tanítási idő nem
  inferencia-mérés.

### Értelmezés (előre)

- Nincs előre rögzített elfogadási küszöb: a különbségeket intervallummal jelentjük, a döntés a felhasználóé.
- Ha a páros különbség intervalluma tartalmazza a 0-t, ezen a mércén nem mutatható ki eltérés; ha nem, az irányt
  és a nagyságot jelentjük.
- A VLM-címke nem ground truth: az Update 1 tanítójele (más filmeken). A fotólétra pontos jel (ismert L), de csak
  egyenletes kameramozgás, egyetlen kódolással.
- A validációs filmek frame-jeit a tanítás epochonkénti kiértékelése látta, de epochválasztásra nem használtuk;
  a tesztfilmeket külön is jelentjük.

### Nincs benne

- A teljes korpusz pontozása (két kiadás egyezése, jelenetkonzisztencia): kb. egy nap GPU-idő, külön döntés.
- A kódolt frame-típus hatása: külön protokoll (R4K-2, a `round4_terv.md` 4b próbája).

## Napló
