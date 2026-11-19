# DATASET INSPECTION REPORT

Dataset: CIC-IDS2017 `GeneratedLabelledFlows` → `TrafficLabelling/` (8 CSV files)
Inspected: 2026-11-19. Every number in sections 1–6 was **measured by script on your files** (not copied from papers). Statements marked *(literature, not verified here)* come from my background knowledge and must be cited/verified before going in the thesis.

---

## 1. Files and size

| File | Rows (real) | Notes |
|---|---:|---|
| Monday-WorkingHours | 529,918 | no attacks; has **second** precision |
| Tuesday-WorkingHours | 445,909 | FTP-Patator, SSH-Patator |
| Wednesday-workingHours | 692,703 | 4 DoS variants + Heartbleed |
| Thursday-Morning-WebAttacks | **170,366** | file has 458,968 rows, but **288,602 are completely empty (all-NaN)** — must be dropped |
| Thursday-Afternoon-Infilteration | 288,602 | Infiltration |
| Friday-Morning | 191,033 | Bot |
| Friday-Afternoon-PortScan | 286,467 | PortScan |
| Friday-Afternoon-DDos | 225,745 | DDoS |
| **Total** | **2,830,743** | 2.83 M real flows, 5 days (3–7 July 2017) |

All files share the **same 85 columns** (same names/order). Labels have leading-space column names in the raw CSV (`" Label"`), so names must be `strip()`ped. Files are Latin-1 encoded, not UTF-8 (the "Web Attack – …" labels contain a byte `0x96`).

## 2. Columns

85 columns = 5 identifier columns (`Flow ID`, `Source IP`, `Source Port`, `Destination IP`, `Destination Port`) + `Protocol` + `Timestamp` + `Label` + 78 CICFlowMeter numeric flow features (durations, packet/byte counts, inter-arrival times, TCP flag counts, window sizes, active/idle times).

Quality problems found:
- `Fwd Header Length.1` is an **exact duplicate** of `Fwd Header Length` (verified on Wednesday) → drop.
- 10 columns are **constant** in Wednesday (e.g. `Bwd PSH Flags`, `Fwd URG Flags`, the six `*Bulk*` columns, `CWE Flag Count`) → will recompute across all days and drop the ones constant everywhere.
- `Flow Bytes/s` and `Flow Packets/s` contain **NaN and ±inf** (zero-duration flows). Total across files: 4,376 inf and 1,338 NaN (excluding the empty rows). → convert inf→NaN, handle with NaN-aware aggregation.
- `Flow Duration` is **negative in 115 rows**, and `Flow IAT Min` negative in a few thousand → measurement bugs; drop negative-duration rows, clip/flag the rest.
- `Init_Win_bytes_forward/backward = -1` in 10⁵–10⁶ rows: this is a "no data" sentinel, not an error → keep but treat as its own signal (or replace by NaN + indicator).
- Exact duplicate rows are rare (≤142 per file after removing empty rows) → negligible.
- `Flow ID` is **not unique** (61.7% repeated) because the same 5-tuple recurs → never use as key.

## 3. Timestamps (most important finding)

Raw examples: `7/7/2017 3:30`, `03/07/2017 08:55:58`.

1. Format is **day/month/year** (`7/7/2017` is ambiguous but `6/7/2017` = 6 July). A naive `pd.to_datetime` parses Monday as 7 March and Thursday-morning as 7 June — **wrong**. Must parse with explicit `%d/%m/%Y`.
2. **Clock is 12-hour with no AM/PM.** Hours in files are 8–12 and 1–5. Hours 1–5 are afternoon. I convert `hour < 8 → +12` and verified that the resulting day spans are contiguous (08:42 → 17:10) and that each afternoon file starts exactly where its morning file ends (PortScan 13:00–15:29 → DDoS 15:30–17:02).
3. **Resolution is 1 minute** in 7 of 8 files (Monday has seconds). So the finest honest time step is **1 minute**. Sub-minute lead times cannot be claimed.
4. Rows are **not time-sorted** in Monday, Tuesday, Wednesday (sorted in the other five) → always sort.
5. `Timestamp` is the flow **start**. A flow only becomes observable when it **ends** (start + `Flow Duration`, µs). Long flows (slowloris, slowhttptest, Hulk hold connections open) therefore are *not* available to a live system at their start minute. Binning by start time would leak the future → see decision D1 in the specification.

Timeline reconstructed from the data (July 2017, 24h clock, minute resolution):

| Day | Window | Attack (flows) active minutes in data |
|---|---|---|
| Mon 3 | 08:55–17:01 | none (pure benign, 487 min) |
| Tue 4 | 08:53–17:00 | FTP-Patator 7,938 flows 09:17–10:30; SSH-Patator 5,897 flows 14:09–15:11 |
| Wed 5 | 08:42–17:10 | slowloris 5,796 (main 09:48–10:11, **stray flows at 09:01 and 14:24–25**); Slowhttptest 5,499 10:15–10:37; Hulk 231,073 10:43–11:07; GoldenEye 10,293 11:10–11:19; Heartbleed 11 flows 15:12–15:32 |
| Thu 6 | 08:59–17:04 | Web-BruteForce 1,507 09:15–10:00; XSS 652 10:15–10:35; SQLi 21 10:40–10:42; Infiltration 36 flows 14:19–15:45 (sporadic) |
| Fri 7 | 08:59–17:02 | Bot 1,966 09:34–12:59; PortScan 158,930 13:05–15:23 (sporadic bursts); DDoS 128,027 15:56–16:16 |

This agrees with the publicly documented CIC-IDS2017 schedule *(literature, not verified here)*, which is strong evidence that my timestamp repair is correct.

## 4. Labels

15 labels (14 attack families + BENIGN):

| Label | Flows | | Label | Flows |
|---|---:|---|---|---:|
| BENIGN | 2,273,097 (80.3%) | | FTP-Patator | 7,938 |
| DoS Hulk | 231,073 | | SSH-Patator | 5,897 |
| PortScan | 158,930 | | DoS slowloris | 5,796 |
| DDoS | 128,027 | | DoS Slowhttptest | 5,499 |
| DoS GoldenEye | 10,293 | | Bot | 1,966 |
| Web-BruteForce | 1,507 | | Web-XSS | 652 |
| Infiltration | 36 | | Web-SQLi 21, Heartbleed 11 | |

Very imbalanced and very uneven: Heartbleed (11 flows), SQLi (21) and Infiltration (36) are far too small to *learn* from; they are only usable as **unseen-family test cases**.

**Label-structure observations that affect the research design**
- Almost every attack is `172.16.0.1 → 192.168.10.50` (one NAT'd attacker, one victim web server). The attacker IP is a perfect shortcut → **IP addresses must never be model features** (leakage by dataset artefact).
- Attack flows are a **small share of traffic in their minute**: median 5.2% of flows in attack-active minutes. Benign background (~800 flows/min median) dilutes whole-network aggregates. Because ports and IPs are forbidden as features, whole-network aggregates may carry no trace of small attacks; this is a real modelling risk.
- Attacks are **concentrated by day** (each day has different families), so a chronological split is also nearly a family split.

## 5. Time-window view (what forecasting actually has to work with)

Aggregating to 1-minute bins gives only **2,454 minutes** in total (487/488/509/486/484 per day), no missing minutes inside any day. **520 minutes** (21%) contain ≥1 attack flow.

Number of **onset events** (v0.2 definition: episodes merged at gap ≤2 min; episodes with <3 active minutes are *tiny bursts*; valid episodes merged into campaigns when separated by <G quiet minutes; onset = first minute of a campaign):

| Setting | Onsets | Per day (Tue/Wed/Thu/Fri) |
|---|---:|---|
| **Primary: min 3 active min, G=10** | **14** | 2 / 2 / 4 / 6 |
| min 2, G=10 | 16 | 2 / 3 / 4 / 7 |
| min 5, G=10 | 12 | 2 / 2 / 3 / 5 |
| min 3, G=5 | 18 | 2 / 4 / 5 / 7 |
| min 3, G=15 | 12 | 2 / 2 / 4 / 4 |
| min 3, G=20 | 11 | 2 / 2 / 3 / 4 |

Primary onsets: Tue FTP-Patator 09:17, SSH-Patator 14:09; Wed DoS campaign 09:48 (slowloris, Slowhttptest, Hulk, GoldenEye chain into **one** campaign), Heartbleed 15:12; Thu Web brute-force 09:15, XSS 10:15, Infiltration 14:28 and 15:12; Fri Bot 10:04, PortScan 13:55 / 14:14 / 14:51 / 15:08, DDoS 15:56. (My earlier looser count of 16–22 counted tiny bursts and per-family fragments as onsets.)

**16 tiny bursts** (masked, not onsets): e.g. slowloris 1 flow at 09:01 and 5 flows at 14:24; PortScan 7 flows 13:05, 1 flow 13:52, 2 flows 15:03; Bot 2 flows 09:34; Hulk tail 145 flows 11:06; and 9 Infiltration fragments of 1–2 flows each.

Benign lead-in before the first attack of each day: Tue 24 min, Wed 19, Thu 16, Fri 35 (Monday supplies 487 attack-free minutes).


## 6. Dataset caveats from the literature *(not verified here — cite before use)*
- Published critiques of CIC-IDS2017 describe CICFlowMeter bugs (TCP-termination handling, flow splitting) and some label errors, which is consistent with the negative durations/NaNs I found.
- It is a lab dataset with a single attacker host per day; results will not automatically transfer to real networks. The thesis must say this.

## 7. FEASIBILITY VERDICT (revised)

**Feasible as an exploratory, event-level study; not as a high-powered statistical study.**
- Supported: causal minute-level stream, explicit onset definition, forward-chaining evaluation, genuine replay, false-alarm measurement.
- Limits: **14 onsets in total**; only Thu (4) and Fri (6) can be test days with supervised training that contains any attacks, so **10 test onsets**; training blocks hold only 2–6 events. Most test families are unseen in training because families cluster by day (drift and novelty are confounded).
- Whole-network aggregates may show no onset signal for low-volume families. No performance can be promised; a null result is a valid outcome.
- Sub-minute lead time is impossible.
- Phase-3 power gate will print exact positives/events per block before any training.

## 8. Is CIC-IDS2017 suitable, or is another dataset better?

Assessment is limited to what I verified (CIC-IDS2017, measured) and my background knowledge of others *(not verified here — I have not downloaded or inspected them; confirm before citing)*.

| Dataset | Fit for onset forecasting |
|---|---|
| **CIC-IDS2017 (in hand)** | Verified: timestamped, labelled campaigns, attack-free Monday, identical feature schema. Weak: 14 onsets, 1-min timestamps, one attacker host, known CICFlowMeter/label issues *(literature)*. |
| CSE-CIC-IDS2018 | Same toolchain and feature schema, many more days and attack campaigns *(from memory)* → the most natural second dataset for replication and more events. Large download; not needed to start. |
| UNSW-NB15 | Synthetic attack mix; I recall little campaign/onset structure and short capture windows → likely poorer for onset forecasting *(low confidence)*. |
| CTU-13 | Botnet-only scenarios, few onsets, different flow format *(from memory)* → poor fit for multi-family study. |
| LANL | Large real-network data whose red-team labels are on authentication events, not attack flows *(from memory)* → different problem. |

**Recommendation [D8]:** keep CIC-IDS2017 as the primary dataset because it is available, verified, reproducible and sufficient for a clearly-scoped exploratory study; do **not** switch now. Keep dataset access behind one adapter so CSE-CIC-IDS2018 can be added as a replication if the Phase-3 power gate shows too few training events or the supervisor wants stronger evidence. State the 14-event limitation prominently in the thesis.
