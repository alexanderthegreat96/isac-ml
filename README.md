# ISAC Neural Net — Anti-Cheat (The Division 2)

A machine-learning project that detects cheaters in **The Division 2** from lifetime
gameplay statistics. The goal of releasing it is to encourage Ubisoft to consider
robust, AI-assisted anti-cheat.

> **Name vs. reality:** the project is called "Neural Net" because versions **v2–v6**
> were Keras neural networks. The current and recommended system, **v8**, is a
> deliberate rethink: a deterministic rule plus a gradient-boosted *review-queue*
> ranker. It is **not** a neural network, and that change is the whole point (see
> [Why v8 exists](#-why-v8-exists-label-circularity)).

---

## 📑 Contents

- [What's current](#-whats-current-v8)
- [Why v8 exists: label circularity](#-why-v8-exists-label-circularity)
- [Architecture](#-architecture-v8)
- [Feature engineering](#-feature-engineering)
- [Research findings & decisions](#-research-findings--decisions)
- [Dataset](#-dataset)
- [Project structure](#-project-structure)
- [Install](#-install)
- [Usage](#-usage)
- [Output format](#-output-format)
- [Model version history](#-model-version-history)
- [IsacCalculator reference](#-isaccalculator-reference)
- [Notes & disclaimer](#-notes--disclaimer)

---

## 🚀 What's current: v8

**v8 is the current, recommended system.** It splits the problem into three parts,
each doing only what it is entitled to do:

| Component | What it is | What it decides |
|---|---|---|
| **Stage 1 — rule** | `headshotsPerHour > 1000`, deterministic | **Automatic flag.** Auditable, no model involved. |
| **Stage 2 — model** | `HistGradientBoostingClassifier` trained only on cheaters the rule never catches | **Review-queue ranking.** Never bans — ranks players for a human. |
| **Cheater scorer** | Whole-dataset, calibrated `P(cheater)` | **Nothing.** A display-only 0–100% number. |

Two files drive everything:

```bash
python3 v8_cheater_detector.py     # inference (run from repo root)
python3 train/v8_train.py          # (re)training
```

---

## 🔬 Why v8 exists: label circularity

The single most important finding from auditing the data:

> **~92% of the `isCheater` labels are reproducible from the rule
> `headshotsPerHour > 800`.**

| Rule | Share of labelled cheaters it explains |
|---|---|
| `hs/h_max > 750` | 93.6% |
| `hs/h_max > 800` | 92.7% |
| `hs/h_max > 850` | 90.9% |

A model trained to predict those labels **cannot beat the rule, because the rule *is*
the label**. v6/v7 reported ~0.98 ROC AUC and that number meant almost nothing — it was
the model rediscovering the threshold it was trained on.

So v8 stops trying to out-classify the rule:

- **Above the ban line** → the rule decides (Stage 1). No model, no circularity.
- **Below the ban line** → a model can add value, because the *only non-circular
  positives* are the **~290 cheaters the rule never fires on**. Stage 2 trains on those
  and produces a **ranked review queue**, not a verdict.

---

## 🧠 Architecture (v8)

### Decision ladder (by damped `headshotsPerHour`)

| `headshotsPerHour` | Outcome | Driven by |
|---|---|---|
| **> 1000** | **Auto-flag** (`RULE_FLAG`, `is_cheater=true`) | Stage-1 rule; model **not** consulted |
| **700 – 1000** | Queued for review (priority `MEDIUM`) | Review-band rule — always, regardless of score |
| **≤ 700** | Model decides: queue / monitor / clear | Stage-2 score vs. queue threshold |

The review decision is **not** a pure function of the model score: a player in the
700–1000 band is queued by the *rule* even if the model scores them low, and a
mid-scoring player below 700 may only be "monitored." The booleans `should_flag` /
`should_review` are the two signals a consumer acts on.

### Stage 2 — the review-queue model

- **Algorithm:** `HistGradientBoostingClassifier` (`class_weight="balanced"`, seeded).
- **Training population:** players **below** the labelling threshold (805 hs/h) only —
  the non-circular positives.
- **Split:** grouped by player `identifier` (`GroupShuffleSplit`) so snapshots of one
  player can't leak across train/test.
- **Honest performance** (needle-in-a-haystack by design):

  | Metric | Value |
  |---|---|
  | Base rate (cheaters in the pop) | **0.27%** |
  | ROC AUC | ~0.88 |
  | PR AUC | ~0.13 (single fold) / ~0.20 (15-seed mean) |
  | Queue threshold | 0.906 |
  | Operating point | ~20% precision @ 36% recall → **~73× lift over base rate** |

  `precision ≥ 0.40` is *unreachable* — expected, and fine. Stage 2 is a **ranker for
  triage**, not a classifier; finding ~50 cheaters per 250 reviewed (vs. <1 by random
  review) is the win.

### Cheater scorer — the display percentage

A separate, **whole-dataset**, isotonic-**calibrated** `P(cheater)`. It trains on *every*
player (rule-flagged included), so its ROC AUC (~0.97) looks great and **means little**
— it mostly restates "is `hs/h` high." It is **well-calibrated** (Brier ≈ 0.0125;
predicted ≈ actual cheater rate per bin) and it **drives no verdict**. It exists so the
API can return an intuitive `cheater_percentage` next to the raw `confidence_score`.

---

## 🧮 Feature engineering

All per-hour rates are damped by a **tiered, playtime-based divisor** (`IsacCalculator`):
`3×` under 100h, decaying to `1×` at 800h+. This suppresses noisy rates from short
sessions — a deliberate trade-off (it can also mask short-playtime cheaters; see below).

**Model features (10):**

```
headshotsPerHour      bodyshotsPerHour      weaponHitsPerHour   npcKillsPerHour
percentageOfHeadshots bsToHsRatio           headshotKillRatio   timePlayed
criticalHitsPerHour   percentageOfCriticalHits
```

- **Inflation-invariant ratios** (reported as such): `percentageOfHeadshots`,
  `percentageOfCriticalHits`, `bsToHsRatio`, `headshotKillRatio`.
- **Report-only** (shown in the deviation report, never fed to the model — absolute
  lifetime counts that scale with playtime): `totalHeadshots`, `totalBodyshots`,
  `totalCriticalHits`.
- **Excluded on purpose:** `headshotKillsPerHour` (algebraically `0.30 × headshots`,
  correlation 0.987 — redundant); the removed **daily** stats (see below).

---

## 📈 Research findings & decisions

1. **Label circularity (~92%)** — the finding that reshaped the whole design. See above.
2. **Daily stats removed from the schema.** `dailyHeadshots/NpcKills/SumCriticalHits`
   were present for ~13% of rows and their *availability* was biased (present for ~39%
   of cheater rows vs. ~12% of legit) — predictive of the label for collection reasons,
   not behaviour. A 15-seed paired test also found no lift (p = 0.012 **against**). Gone.
3. **`criticalHits` added — and honestly, it's marginal.** Paired ablation:
   `base` PR AUC 0.194 vs. `base + crit` 0.200 (**+0.006, wins 8/15 seeds**) — inside
   the noise. Kept because it's a *real* behavioural signal (unlike daily availability)
   and doesn't hurt. Drop it by setting `LIFETIME_FEATS = BASE_FEATS`. Re-check anytime
   with `python3 train/v8_train.py --ablate`.
4. **`headshotKillsPerHour` removed** — it simplifies to `0.30 × headshots`, correlating
   0.987 with `headshotsPerHour`. Same variable rescaled.
5. **Deviation baseline is legit-players-only.** Reference medians are computed on
   `isCheater == 0` in the population, fixing a v6 bug where the baseline included
   (upsampled) cheaters and understated every deviation.
6. **Divisor damping trade-off.** `tiered` reproduces production; a `smooth` mode exists
   that removes the step discontinuities and behaves better above ~85% recall. Set via
   `DIVISOR_MODE` in `train/v8_train.py`.
7. **Reproducible & platform-independent.** Fully seeded — a Linux retrain reproduces the
   Windows models bit-for-bit (identical scores, thresholds, and stats), provided the
   **same scikit-learn version** (1.9.0, recorded in the manifest) is used.

---

## 📦 Dataset

`train/isac-ml-training-dataset.csv` — **178,543 rows**, **3.84% cheaters**, curated so
no `headshots > weaponHits`, `criticalHits > weaponHits`, `timePlayed ≤ 0`, or
`weaponHits ≤ 0` remain (the trainer keeps a fail-loud guard regardless).

**Schema:**

```
username, identifier, timePlayed, weaponHits, criticalHits, headshots, bodyshots,
headshotKills, npcKills, headshotsPerHour, bodyshotsPerHour, weaponHitsPerHour,
isCheater, created_at
```

`criticalHits` is the subset of `weaponHits` that landed a critical, so it is always
≤ `weaponHits`.

---

## 📂 Project structure

```
v8_cheater_detector.py       # v8 inference (run from repo root) — CURRENT
v1..v6_cheater_detector.py   # legacy detectors (v2–v6 are Keras NNs)

train/
    v8_train.py              # v8 training: rule audit, Stage-2 model, cheater scorer
    calc.py                  # IsacCalculator — per-hour scaling & ratios
    isac-ml-training-dataset.csv
    vX_neural_net_train.py   # legacy v2–v6 training scripts
    curate.py, run.py, train.py, neural_net_train.py   # legacy/exploratory

models/
    v8_manifest.json         # thresholds, feature lists, model registry, checksums
    v8_stage2_stage2.pkl     # Stage-2 review-queue model      (git-ignored)
    v8_stats_stage2.json     # legit-baseline stats for the deviation report
    v8_cheater_scorer.pkl    # calibrated display scorer        (git-ignored)
    vX_feature_*.json        # legacy v2–v6 feature statistics
```

> **Note:** `*.pkl`/`*.keras` are git-ignored, so trained binaries are not committed —
> retrain to regenerate them. The manifest and `*_stats_*.json` files *are* tracked.

---

## ⚙️ Install

v8 needs only the scikit-learn stack (no TensorFlow — that's for the legacy v2–v6 NNs):

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -U pip
pip install numpy pandas joblib matplotlib scikit-learn==1.9.0
# or, for the legacy versions too:  pip install -r requirements.txt
```

Pin **scikit-learn 1.9.0** (the version in the manifest) so trained pickles load without
warnings. Python 3.9+.

---

## 🏃 Usage

Run everything **from the repo root** (paths are relative: it reads
`train/isac-ml-training-dataset.csv` and writes to `models/`).

**Detection:**

```bash
python3 v8_cheater_detector.py
```

**Training** (regenerates `models/v8_*`):

```bash
python3 train/v8_train.py                    # train Stage-2 model + cheater scorer
python3 train/v8_train.py --ablate           # + measure whether criticalHits helps
python3 train/v8_train.py --data path.csv    # custom dataset
```

**Programmatic:**

```python
from v8_cheater_detector import IsacAntiCheat

detector = IsacAntiCheat("models")
result = detector.predict(dict(
    timePlayed=803, weaponHits=3930785, criticalHits=1572000,
    headshots=750885, headshotKills=225266, npcKills=192361,
))
```

---

## 📤 Output format

Every response is wrapped in a `{status, code, data}` envelope. `status`/`code` reflect
whether the input could be **scored** (a confirmed cheater is still `200/true`; only an
unscorable input is `422/false`).

```jsonc
{
  "status": true, "code": 200,
  "data": {
    "is_cheater": true, "should_flag": true, "should_review": false,
    "classification_tier": "Confirmed cheater",
    "cheater_probability": "100.00%",   // calibrated scorer, as a string
    "cheater_percentage": 100.0,        //   ""            , as a number
    "confidence_score": 1.0,            // the QUEUE model's confidence (0..1), NOT a cheat probability
    "verdict": "RULE_FLAG", "action": "automatic", "stage": 1,
    "rule": "headshotsPerHour > 1000.0", "margin_over_threshold": 2667,
    "reason": "...", "note": "...",
    "advisory": {                       // model scores that did NOT decide (on rule flags)
      "advisory_only": true, "model_population": "stage2",
      "score": 0.6657, "queue_threshold": 0.9057,
      "would_queue_independently": false, "cheater_percentage": 90.36,
      "caveat": "Circular: the rule generated these models' labels..."
    },
    "computed": { /* per-feature deviation report — ALWAYS present on stage 1 & 2 */ }
  }
}
```

**Key fields:**

- **`should_flag` / `should_review`** — the two actionable booleans (never both true).
- **`confidence_score`** — the Stage-2 *queue* model's confidence (a ranking score, not a
  cheat probability). `1.0` on a rule flag.
- **`cheater_probability` / `cheater_percentage`** — the calibrated whole-dataset
  `P(cheater)`. The "how likely a cheater" number. Pinned to 100% on a rule flag.
- **`advisory`** — on a rule flag, the model scores are surfaced here, flagged
  `advisory_only` with a circularity caveat, so a reviewer sees the model's opinion
  without it ever influencing the verdict.
- **`computed`** — a per-feature deviation report (input vs. legit-player median, MAD /
  std z-scores, %-deviation, p99, and an `Extreme/Strong/Mild/Normal` flag). This is the
  appeal-facing artifact; it is always returned for scorable inputs.

---

## 🕰️ Model version history

| Version | Approach | Status |
|---|---|---|
| v1 | Pickled baseline classifier | legacy |
| v2–v5 | Keras NNs, growing feature engineering (medians/MADs/stds) | legacy |
| v6 | Keras NN, 17 features, z/MAD deviation report | legacy (was "production") |
| **v8** | **Rule (Stage 1) + gradient-boosted review queue (Stage 2) + calibrated display scorer** | **current / recommended** |

There is no v7. v8's ~0.13–0.20 Stage-2 PR AUC is a *smaller, honest* number that
replaces v6's misleading 0.98.

---

## 📘 IsacCalculator reference

```python
from train.calc import IsacCalculator   # or `from calc import IsacCalculator` inside train/
```

The `IsacCalculator` turns raw lifetime stats into **damped per-hour metrics** and
ratios used both for the deviation report and as model features.

**Initialization:**

```python
calc = IsacCalculator(
    time_played_total=8940,   # hours (or via time_unit_convert_to_hours)
    headshots=2565068,
    sum_hits=37037181,        # weaponHits
    sum_critical_hits=500000, # criticalHits
    kills_npc=454610,
    kills_headshot=769521,
)
```

**Per-hour metrics** (tiered-divisor scaled): `calculate_headshots_per_hour()`,
`calculate_bodyshots_per_hour()`, `calculate_weapon_hits_per_hour()`,
`calculate_critical_hits_per_hour()`, `calculate_npc_kills_per_hour()`,
`calculate_headshot_kills_per_hour()`. Each returns `0` when the raw stat is too small to
scale safely.

**Ratios** — `calculate_percentages_and_ratios()`:

```json
{
  "percentageOfHeadshots": 12.34,
  "percentageOfBodyshots": 87.66,
  "totalHeadshots": 2565068,
  "totalBodyshots": 34472113,
  "hsToBsRatio": 0.07,
  "bsToHsRatio": 13.44
}
```

**Time parsing** — `time_unit_convert_to_hours("2h30m")` → `3` (rounded hours).

Notes:
- All scaling is divisor-based to prevent anomalies in short play sessions.
- Extreme values are handled gracefully (floored division, capped deviations).
- Outputs feed the **v8** detector (`criticalHits` is now part of the feature set).

---

## 📌 Notes & disclaimer

- Provided for **research and demonstration purposes only**. Not affiliated with or
  endorsed by Ubisoft.
- The models label a player by comparing behaviour to a legitimate baseline and to a
  deterministic rule; Stage 2 **ranks for human review**, it does not establish cheating.
- The Division 2 is a trademark of Ubisoft. Respect the terms of service of any game you
  use this with.
