# Precipitation-event catalog

`python src/run.py events` produces `events/`: the top **5 snow, 5 rain and 5 mixed** events
over New York City in the OpenMesh period (29 Oct 2023 – 1 Jul 2024), each with its own folder.
Implementation: `nyc_rain_maps.events`.

## 1. Hourly record

MRMS `MultiSensor_QPE_01H_Pass2` for every hour of the period over the NYC domain
(40.48–40.93 N, 74.27–73.68 W, five boroughs plus margin; 45 × 59 cells). The event signal is
the domain-mean hourly accumulation (hours with < 80 % valid cells are treated as missing).

## 2. Detection

- wet hour: domain mean ≥ 0.1 mm;
- wet hours separated by fewer than 6 dry hours belong to one event (the usual
  inter-event time for urban hydrology);
- events with a domain-mean total < 1 mm are discarded.

Each event records its window (start of the first wet hour → end of the last), duration, wet
hours, domain-mean total, largest cell total, peak hour and peak rates.

## 3. Precipitation type — two independent sources

**MRMS PrecipFlag** (model temperature profile + radar vertical structure), sampled every 10
minutes during the event's wet hours. Each flagged cell is weighted by that hour's QPE in the
cell, so the result is the fraction of the event's *precipitation* (not area) that fell as snow:

    snow_fraction = Σ QPE·[flag = snow] / Σ QPE·[flag ∈ snow ∪ rain flags]

**ASOS present weather** at Central Park (NYC), LaGuardia, JFK and Newark (IEM METAR archive):
each station-hour is typed from its `wxcodes` as rain (RA, DZ), snow (SN, SG), mix (PL, GS or
snow with rain), freezing (FZRA, FZDZ) or none.

**Rules** — both sources must agree for a "pure" type:

| type | rule |
|---|---|
| snow | snow fraction ≥ 0.75 **and** < 25 % of ASOS precipitation station-hours report anything other than snow |
| rain | snow fraction ≤ 0.05 **and** no *material* ASOS snow, mix or freezing reports (fewer than 2 station-hours or < 5 % of precipitation station-hours — a lone sleet observation in a 100-station-hour rainstorm does not make it mixed) |
| mix | everything else (both phases, freezing rain, or the two sources disagree) |

Warm events (minimum ASOS temperature ≥ 6 °C and no frozen report) are rain without
downloading PrecipFlag: snow is physically excluded at those surface temperatures.

Example of why both sources are used: on 16 Jan 2024 MRMS flags 97 % of the precipitation as
snow, but JFK and Newark reported freezing rain for 27 station-hours — the event is `mix`.

## 4. Filling snow and mix from other winters

2023–24 gave only two snow events and three mixed events over the city. When a class has fewer
than five, the same method is run on other winters (Nov 2020 – Apr 2026, the MRMS Pass 2 archive
on AWS), screening cheaply first:

1. download ASOS present weather for the whole extended period (one small CSV);
2. keep UTC days with ≥ 3 station-hours of snow, mix or freezing reports;
3. group those days into clusters (gaps ≤ 2 days), fetch hourly MRMS only for each cluster ±1 day;
4. detect events, classify them exactly as above, keep the snow and mix ones.

These events have radar and ASOS analysis but **no link data**, so no CML maps; the catalog marks
them *radar only* and ranks them after the OpenMesh-period events of the same class.

## 5. Ranking

Within each type, events inside the OpenMesh record come first, then radar-only ones; each group
is ranked by domain-mean total (liquid equivalent), and the top five of each type are marked
`selected`. `events/all_detected_events.csv` lists every detected event with its evidence
(snow fraction, ASOS present-weather counts and totals, minimum temperature). One folder per
selected event, with figures and per-event scores, is kept in
the `pcpn_maps` repository (`drorjac/pcpn_maps`, private; `events/`).

## Caveats

- Snow totals are liquid-water equivalent. Radar QPE in snow is much less certain than in rain;
  CMLs barely respond to dry snow and over-respond to wet snow (melting layer). Snow and mix
  events are in the catalog to test the methods outside their design range, not as a
  validation of snow measurement.
- 2023–24 was a historically snow-poor winter in New York City (Central Park's first
  measurable snow in ~700 days fell on 16 Jan 2024), so the snow class may contain fewer than
  five events or only light ones; the README says so when that happens rather than relaxing
  the definition.
