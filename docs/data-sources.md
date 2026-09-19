# CoH1 data sources

Survey done 2026-09-19. "Verified" = fetched during the survey. Target data version:
2.602 (2.700 is reported to be a Steamworks migration without balance changes; not
verified).

## Primary: coh-stats.com mirror (verified live)

https://www.hq-coh.com/stats/coh-stats.com/Main_Page.html

Static HTML copy (Dec 2013) of a wiki bot-generated from the game's RGD attribute
files. ~400 pages with consistent tables; index pages follow
`Category_American_Weapons.html`, `Category_Wehrmacht_Structures.html`, etc.
Data is ~2.600/2.601 and must be patched forward with the 2.602 notes below.

Covers: squads (size, squad HP, cost, pop, build time, upkeep, sight, capture rate,
reinforce and retreat modifiers, suppression thresholds, veterancy), vehicles (HP, cost,
speed, accel, rotation, target type), weapons (damage, range bands, accuracy per band,
penetration per band, suppression per band, reload/cooldown/burst, moving accuracy,
nearby suppression, AOE, deflection damage, **per-weapon cover table**, **per-weapon
target table** of ~55 target types with accuracy / moving / damage / penetration /
rear-penetration / suppression / priority), structures (cost, time, HP, products),
upgrades, and `Info_Suppression.html` (thresholds, recovery, pinned effects, cover
recovery multipliers).

Missing: infantry move speed, all resource/territory/VP data. Per-model HP must be
derived as squad HP / squad size. When scraping, note costs are separated by resource
icons in the HTML, not text.

Verified samples (pre-2.602):

- Riflemen: 6 men, 330 HP, 270 MP, pop 6, 36 s, upkeep 14.4 MP/min, sight 35, capture
  ×1.5; suppress 0.2 / pin 0.6 / recovery 0.008.
- M1 Garand: damage 10; ranges 8/17/35; accuracy 0.75/0.55/0.35; moving ×0.5.
- MG42 HMG team: 3 men, 165 HP, 260 MP, 40 s. MG42: damage 7; ranges 11/22/45;
  accuracy 0.6/0.3/0.125; suppression 0.015/0.015/0.0125; setup 3 s.
- Sherman: 636 HP, 420 MP / 90 F, 55 s, pop 8, speed 5.2, rotation 35. 75 mm: damage
  87.5; ranges 10/20/40; accuracy 1/1/0.75; penetration 1/0.92/0.83; reload 6 s;
  deflection damage ×0.15.
- Small-arms cover (accuracy / damage / suppression): light 0.5/1/0.5; heavy
  0.5/0.5/0.1; negative 1.25/1.25/1.5; garrison 0.4/0.75/0.
- Standard infantry suppression: activate 0.2, recover 0.15, pin 0.6, pin-recover 0.5,
  recovery 0.008/s; 7 s out-of-combat delay then ×50; cover recovery multipliers
  light 2.5, heavy 5, negative 0.5.

## Patch-forward: official 2.602 release notes (verified live)

https://www.gamereplays.org/community/Official_CoH_2602_Release_Notes-t782742.html

Relevant deltas include MG42 team 260→250 MP and 40→35 s; Pak 38 310→290 MP; Engineer
suppression brought in line with other infantry; US mortar and .30 cal setup/pack-up
times; 57 mm reinforce cost and accuracy vs infantry; Hellcat speed 6→7.2; Pershing and
Tiger 900→800 MP; StuG accel/decel.

## Cross-check / schema: omgmod JSON dump (verified live, modded)

https://github.com/omgmod/omg_rails — `lib/assets/stats/{weapons,units,entities,upgrades}.json`

Machine-readable, Relic attribute key names. Core weapon values spot-checked identical
to coh-stats (Garand, MG42, Kar98k, Sherman 75 mm). Squads, costs and economy are
modded — use only for weapons and entity fields (e.g. infantry `speed_max`,
Volksgrenadier = 3). Field semantics: https://github.com/omgmod/attrib-documentation.
Lua→JSON converter: https://github.com/omgmod/attrib-parser.

## Economy, territory, maps (weakest area)

- companyofheroes.fandom.com (use `api.php?action=parse&prop=wikitext`): +5 fuel/min HQ
  income; 500 VP default; supply cut-off rules; point income low 5 / med 10 / high 16;
  Supply Yard costs and upkeep reduction to 75% / 54% / 33%; tech prerequisites.
  `Tactical_Maps` page has top-down PNGs for Angoville, Semois, Langres, Wrecked Train,
  Rails and Metal, Vire River Valley (download needs a browser; content not inspected).
- Steam "Full Knowledge Guide" (id 2876755869): OP income +5→+8, +10→+16, +16→+24; OP
  cost 200 MP.
- ACGIM README: base pop cap 30; sector pop +2 strategic, +3/+5/+7 low/med/high.
- Unconfirmed (search snippets only): +3 MP/min per held point; VP ticker every 4 s,
  drain proportional to VP lead.

## Known gaps → `estimated` until extracted

Base manpower income; how upkeep combines with income; capture/decapture times per
point type; VP ticker rate; secured medium-point income (16 vs 18); most infantry move
speeds; map sector graphs and coordinates (we author our own maps anyway).

## Exact route (optional, needs a Steam copy of CoH1)

Open the attrib SGA archives in Corsix Mod Studio, dump RGDs to Lua, convert with
omgmod/attrib-parser. Yields exact 2.700 data including strategic-point entities
(income, pop, capture time). Archive path and VP-ticker script location are
unverified recollections.
