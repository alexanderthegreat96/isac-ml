# ------------------------------------------------------------------------------
# ISAC cheat detection - v8
#
# WHAT CHANGED AND WHY
#
# The single most important finding from the data audit: your isCheater labels
# are ~92.5% reproducible from `headshotsPerHour > 800` at player level. A model
# trained to predict those labels cannot beat the rule, because the rule IS the
# label. v6/v7 measured 0.98 AUC and that number meant almost nothing.
#
# So v8 stops trying to out-classify the rule and splits the problem in two:
#
#   STAGE 1 (rule)  - players above the ban threshold. Deterministic, auditable,
#                     unchanged in spirit from production. No model involved.
#   STAGE 2 (model) - players the rule CLEARS. Trained on the ~300 cheaters that
#                     the rule never fires on - the only non-circular positives
#                     in the dataset. Outputs a ranked REVIEW QUEUE, not a ban.
#
# Stage 2 is where a model can add something, because those players are labelled
# by a process independent of the threshold.
#
# CHEATER PERCENTAGE - a whole-dataset calibrated score, for DISPLAY only
# Separate from stage 2: cheater_scorer trains on EVERY player (rule-flagged
# included) to predict isCheater, then isotonic-calibrates to real prevalence.
# Because the labels are ~92% the hs/h rule, this number mostly restates "is
# hs/h high" - its ROC AUC looks great and means little. It drives NOTHING; the
# verdict and queue still come from the rule + stage 2. It exists so the API can
# return a smooth 0-100% cheater_percentage next to the raw confidence score.
#
# DAILY STATS - REMOVED from the schema entirely
# dailyHeadshots / dailyNpcKills / dailySumCriticalHits were present for only
# ~13% of rows, and availability was heavily biased (39% of cheater rows had them
# vs 12% of legit), so availability predicted the LABEL for collection reasons,
# not behaviour. A 15-seed paired test also found the features did not help
# (p=0.012 AGAINST inclusion). They are gone from the dataset now, and with them
# the two-population (has_daily) routing: STAGE 2 IS NOW A SINGLE MODEL.
#
# CRITICAL HITS - new lifetime field
# criticalHits is the subset of weaponHits that landed a critical, so it is
# always <= weaponHits. It contributes two features: criticalHitsPerHour (a
# damped rate) and percentageOfCriticalHits = 100*crit/weaponHits (a crit-share
# ratio, invariant to the inflation proc). totalCriticalHits is reported for
# context only. Whether crit actually separates cheaters is MEASURED, not
# assumed - run --ablate to compare the queue with and without CRIT_FEATS.
#
# DATA INTEGRITY - now curated upstream
# The dataset no longer contains headshots > weaponHits, criticalHits >
# weaponHits, timePlayed <= 0 or weaponHits <= 0. The filter below is kept as a
# DEFENSIVE GUARD: it should drop nothing, and warns loudly if a future refresh
# regresses.
#
# DERIVED COLUMNS - headshotKills is not a measurement
# calculate_total_headshot_kills() simplifies algebraically to headshots * 0.30:
# total_kills and avg_bullets_to_kill both cancel. 94.6% of rows match exactly.
# So headshotKillsPerHour correlates 0.987 with headshotsPerHour - the same
# variable rescaled. It is REMOVED from the feature set. hsKillRatio is kept
# (it is really headshots/npcKills, correlation with the rate only 0.33) but
# renamed so nobody reads it as a kill-quality signal.
#
# INFLATION
# A ~10% per-shot double-damage proc inflates counters multiplicatively. Above
# ~10k shots the factor is effectively constant (relative sd < 0.3%), so it is
# corrected by division, not by statistics. Set INFLATION_FACTOR once confirmed.
# Ratio features (pctHeadshots, bsToHs, hsKillRatio) are invariant to it and are
# weighted accordingly.
# ------------------------------------------------------------------------------

import os
import json
import hashlib
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (precision_recall_curve, roc_auc_score,
                             average_precision_score, classification_report,
                             brier_score_loss)

# --- Detection constants ----------------------------------------------------
# Graduated bands instead of one cliff:
#   > AUTO_BAN_THRESHOLD          -> stage 1, automatic flag
#   REVIEW_BAND_FLOOR..AUTO_BAN   -> queued for review, priority from stage-2 score
#   < REVIEW_BAND_FLOOR           -> stage 2 decides queue/clear as before
# LABEL_RULE_THRESHOLD is where the historical labelling rule sat. Stage 2
# trains ONLY below it: players above carry labels the rule generated, and
# training on them would teach the model the threshold all over again.
AUTO_BAN_THRESHOLD   = 1000.0
REVIEW_BAND_FLOOR    = 700.0
LABEL_RULE_THRESHOLD = 805.0
QUEUE_FLOOR          = 700.0   # kept for manifest back-compat
INFLATION_FACTOR = 1.00    # set to 1.10 once the double-damage proc is confirmed
# "tiered" reproduces production exactly. "smooth" removes the step
# discontinuities (up to +33% score jump for one extra hour of play) while
# keeping the same damping intent. Measured at matched recall: tiered cuts
# false positives 10-14% vs undamped below 85% recall, but costs +83% at 90%.
# smooth tracks tiered below 85% and is far better above it.
DIVISOR_MODE = "tiered"    # "tiered" | "smooth"
SMOOTH_K = 200.0
RECALL_CROSSOVER = 0.85    # above this, damping starts adding false positives

# --- Config -----------------------------------------------------------------
SEED       = 42
TEST_SIZE  = 0.30
OUT_DIR    = "models"
VERSION    = "v8"
MIN_POSITIVES = 30         # below this a fold cannot support a model

# BASE_FEATS are the lifetime signals carried from v6-v8. CRIT_FEATS are the new
# criticalHits-derived pair, kept separate so --ablate can measure their lift
# before they earn a permanent place.
BASE_FEATS = [
    "headshotsPerHour", "bodyshotsPerHour", "weaponHitsPerHour",
    "npcKillsPerHour",
    "percentageOfHeadshots", "bsToHsRatio", "headshotKillRatio", "timePlayed",
]
CRIT_FEATS = ["criticalHitsPerHour", "percentageOfCriticalHits"]
LIFETIME_FEATS = BASE_FEATS + CRIT_FEATS
# headshotKillsPerHour deliberately excluded - see header (corr 0.987).
# Ratio features are inflation-invariant; flagged so they can be reported on.
INVARIANT_FEATS = ["percentageOfHeadshots", "percentageOfCriticalHits",
                   "bsToHsRatio", "headshotKillRatio"]

# Report-only columns: they get a deviation row in the appeal-facing report but
# are NEVER fed to the model. These are absolute LIFETIME counts, not per-hour
# rates, so they are NOT playtime-normalised - a high-playtime legit player will
# deviate strongly on them purely from hours played. They exist to give a human
# reviewer raw context ("2.5M headshots vs a typical 200k"); do not drive
# verdicts off them, and read their flag alongside timePlayed.
REPORT_ONLY_FEATS = ["totalHeadshots", "totalBodyshots", "totalCriticalHits"]


# ---------------------------------------------------------------------------
def load_and_prepare(path):
    df = pd.read_csv(path)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", format="mixed")
    n0 = len(df)

    # Integrity guard. The dataset is curated upstream, so every condition below
    # should be empty. It is kept as a fail-loud tripwire: criticalHits and
    # headshots are both SUBSETS of weaponHits (a crit is a hit that critted, a
    # headshot a hit that hit the head), so either exceeding weaponHits means
    # that row's weaponHits failed to populate and its ratios (bsToHs,
    # pctHeadshots, pctCrit) would be garbage.
    bad = ((df.headshots > df.weaponHits) | (df.criticalHits > df.weaponHits)
           | (df.timePlayed <= 0) | (df.weaponHits <= 0))
    if bad.any():
        print(f"[WARN] integrity guard dropped {int(bad.sum())} impossible rows the "
              f"curation should have removed - check the upstream export.")
        df = df[~bad].copy()
    else:
        print(f"[INFO] integrity guard: clean, 0 impossible rows in {n0} (as expected)")

    if INFLATION_FACTOR != 1.0:
        for c in ["weaponHits", "criticalHits", "headshots", "bodyshots", "headshotKills", "npcKills"]:
            df[c] = df[c] / INFLATION_FACTOR
        print(f"[INFO] counters divided by {INFLATION_FACTOR} to undo proc inflation")

    tp = df.timePlayed.astype(float)
    if DIVISOR_MODE == "smooth":
        # continuous analogue of the tiered multiplier: 3x near zero decaying
        # to 1x, monotonic, no boundary jumps.
        div = tp * (1 + 2*SMOOTH_K/(tp + SMOOTH_K))
    else:
        mult = np.select([tp<100, tp<200, tp<300, tp<400, tp<600, tp<800],
                         [3.0, 2.5, 2.0, 1.5, 1.2, 1.1], default=1.0)
        div = tp * mult
    print(f"[INFO] divisor mode: {DIVISOR_MODE}")
    hs, wh = df.headshots.astype(float), df.weaponHits.astype(float)
    bs = np.maximum(wh - hs, 0)

    def scaled(n):
        return np.where(div > 0, np.floor(np.divide(n, div, out=np.zeros(len(df)), where=div>0)), 0)

    df["headshotsPerHour"]     = np.where(hs > 0, scaled(hs), 0)
    df["bodyshotsPerHour"]     = np.where(bs > tp, scaled(bs), 0)
    df["weaponHitsPerHour"]    = np.where(wh > tp, scaled(wh), 0)
    df["npcKillsPerHour"]      = np.where(df.npcKills > 0, scaled(df.npcKills.astype(float)), 0)
    df["headshotKillsPerHour"] = np.where(df.headshotKills > 0, scaled(df.headshotKills.astype(float)), 0)

    df["percentageOfHeadshots"] = 100 * hs / wh
    df["percentageOfBodyshots"] = 100 * bs / wh
    df["totalHeadshots"] = hs
    df["totalBodyshots"] = bs
    df["hsToBsRatio"] = np.where(bs > 0, hs / bs, 999.0)
    # bs/hs: zero bodyshots with headshots present is the most suspicious pattern
    # and must resolve to 0.0, not NaN. v6 let it fall through fillna(0) via inf.
    df["bsToHsRatio"]       = np.where(hs > 0, bs / hs, 999.0)
    # named for what it is: headshots per NPC kill. headshotKills is 0.30*headshots,
    # so the constant is absorbed and this measures headshot economy, not kill quality.
    df["headshotKillRatio"] = df.headshotKills / (df.npcKills + 1)

    # criticalHits: subset of weaponHits (a hit that critted). The rate is damped
    # like the other per-hour signals; the crit SHARE is a ratio and so is
    # invariant to the inflation proc. totalCriticalHits is report-only context.
    cc = df.criticalHits.astype(float)
    df["criticalHitsPerHour"]      = np.where(cc > tp, scaled(cc), 0)
    df["percentageOfCriticalHits"] = 100 * cc / wh
    df["totalCriticalHits"]        = cc

    df = df.replace([np.inf, -np.inf], np.nan)
    df[LIFETIME_FEATS + REPORT_ONLY_FEATS] = df[LIFETIME_FEATS + REPORT_ONLY_FEATS].fillna(0)
    return df


def audit_labels(df):
    """Quantify how much of the label is just the rule. Governs how much to trust
    any metric produced downstream."""
    p = df.groupby("identifier").agg(hsmax=("headshotsPerHour","max"),
                                     cheat=("isCheater","max"))
    print("\n[AUDIT] label circularity")
    for t in [750, 800, 850]:
        m = p.hsmax > t
        tp_ = int((m & (p.cheat==1)).sum()); fn_ = int((~m & (p.cheat==1)).sum())
        print(f"  hs/h_max>{t}: explains {tp_/(tp_+fn_):.1%} of labelled cheaters")
    resid = int(((p.hsmax <= 800) & (p.cheat==1)).sum())
    print(f"  cheaters the rule NEVER fires on: {resid}  <- the only non-circular positives")
    if resid < MIN_POSITIVES * 2:
        print(f"  [WARN] only {resid} non-circular positives. Stage 2 will be unstable.")
    return p


def fit_stage2(d, feats, name, out):
    """Train the review-queue model on the below-threshold population. Split is
    grouped by identifier: snapshots of one player are highly correlated, so a
    row-level split leaks and inflates every metric."""
    print(f"\n[STAGE 2 :: {name}]")
    print(f"  rows={len(d)}  players={d.identifier.nunique()}  cheater rows={int(d.isCheater.sum())}")
    if d.isCheater.sum() < MIN_POSITIVES:
        print(f"  SKIPPED - fewer than {MIN_POSITIVES} positives.")
        return None

    X = d[feats]; y = d.isCheater.values; g = d.identifier.values
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                    random_state=SEED).split(X, y, g))
    assert not (set(g[tr]) & set(g[te])), "player appears in both folds"
    if y[te].sum() < 10:
        print("  SKIPPED - too few positives in the test fold to evaluate honestly.")
        return None

    clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06,
                                         class_weight="balanced", random_state=SEED)
    clf.fit(X.iloc[tr], y[tr])
    proba = clf.predict_proba(X.iloc[te])[:, 1]

    base = y[te].mean()
    ap = average_precision_score(y[te], proba)
    print(f"  test rows={len(te)}  positives={int(y[te].sum())}  base rate={base:.4%}")
    print(f"  ROC AUC={roc_auc_score(y[te], proba):.4f}  PR AUC={ap:.4f}  lift over base={ap/base:.1f}x")

    prec, rec, thr = precision_recall_curve(y[te], proba)
    chosen = None
    print("  queue operating points:")
    for target in [0.05, 0.10, 0.20, 0.40]:
        ok = np.where(prec[:-1] >= target)[0]
        if len(ok):
            i = ok[np.argmax(rec[:-1][ok])]
            print(f"    precision>={target:.2f}: recall={rec[i]:.3f} thr={thr[i]:.4f} "
                  f"-> {int(rec[i]*y[te].sum())} found per {int(rec[i]*y[te].sum()/target)} reviewed")
            if target == 0.20:
                chosen = float(thr[i])
        else:
            print(f"    precision>={target:.2f}: unreachable")
    if chosen is None:
        chosen = 0.5
        print("  [WARN] 20% precision unreachable; defaulting threshold to 0.5")

    model_path = f"{out}/{VERSION}_stage2_{name}.pkl"
    joblib.dump(clf, model_path)

    # Reference stats for the inference-time deviation report. Computed on LEGIT
    # players in this population only: the report answers "how far is this player
    # from a normal player", so cheaters must not be in the baseline. v6 computed
    # these on the upsampled training frame, which pulled the medians toward
    # cheaters and understated every deviation.
    # Model features PLUS report-only columns (absolute headshots/bodyshots).
    # The extra columns get reference stats for the deviation report but are not
    # in `feats`, so they never reach the classifier - see REPORT_ONLY_FEATS.
    report_cols = feats + REPORT_ONLY_FEATS
    ref = d[d.isCheater == 0][report_cols]
    stats = {}
    for c in report_cols:
        col = ref[c]
        med = float(col.median())
        mad = float(np.median(np.abs(col - med)))
        binary = set(col.dropna().unique()) <= {0, 1}
        stats[c] = {"median": med,
                    "std": float(col.std()) or 1e-6,
                    "mad": (1.0 if binary else (mad if mad else 1e-6)),
                    "binary": bool(binary),
                    "p99": float(col.quantile(0.99))}
    stats_path = f"{out}/{VERSION}_stats_{name}.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Checksums let the detector distinguish "corrupted in transfer" from every
    # other failure. A single flipped byte in a pickle produces errors that name
    # modules nothing imports; this turns that into an immediate, plain message.
    def _sha(path):
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()

    return {"name": name, "features": feats, "threshold": chosen,
            "pr_auc": float(ap), "base_rate": float(base),
            "model_file": f"{VERSION}_stage2_{name}.pkl",
            "stats_file": f"{VERSION}_stats_{name}.json",
            "model_sha256": _sha(model_path),
            "stats_sha256": _sha(stats_path)}


def crit_ablation(sub, n_seeds=15):
    """One fold cannot settle whether the criticalHits features help when the
    test set holds only tens of positives. Repeat across seeds and compare
    paired - exactly how the daily features were vetted before removal."""
    print(f"\n[ABLATION] criticalHits features, {n_seeds} seeds, same players")
    if sub.isCheater.sum() < MIN_POSITIVES:
        print("  SKIPPED - too few positives."); return
    y = sub.isCheater.values; g = sub.identifier.values
    res = {"base only": [], "base + crit": []}
    for s_ in range(n_seeds):
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                        random_state=s_).split(sub, y, g))
        if y[te].sum() < 10:
            continue
        for name, fs in [("base only", BASE_FEATS),
                         ("base + crit", BASE_FEATS + CRIT_FEATS)]:
            c = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06,
                                               class_weight="balanced", random_state=s_)
            c.fit(sub[fs].iloc[tr], y[tr])
            res[name].append(average_precision_score(y[te], c.predict_proba(sub[fs].iloc[te])[:, 1]))
    for k, v in res.items():
        v = np.array(v)
        print(f"  {k:<12} PR AUC mean={v.mean():.4f} sd={v.std():.4f} n={len(v)}")
    a = np.array(res["base only"]); b = np.array(res["base + crit"])
    d = b - a
    print(f"  paired diff (crit - base): {d.mean():+.4f}   crit wins {int((d>0).sum())}/{len(d)} seeds")
    print("  -> drop CRIT_FEATS from LIFETIME_FEATS if this is not consistently positive.")


def fit_cheater_scorer(df, feats, out):
    """Whole-population, calibrated P(isCheater) for DISPLAY (see header). Grouped
    split so the reported AUC is not leaked; isotonic calibration so the output
    reads as a real percentage. Trains on ALL players, unlike stage 2 - so it is
    deliberately circular and decides nothing."""
    print("\n[CHEATER SCORER] whole-dataset calibrated P(isCheater) - display only")
    X = df[feats]; y = df.isCheater.values; g = df.identifier.values
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                    random_state=SEED).split(X, y, g))
    assert not (set(g[tr]) & set(g[te])), "player appears in both folds"
    base = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                          random_state=SEED)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(X.iloc[tr], y[tr])
    proba = clf.predict_proba(X.iloc[te])[:, 1]
    print(f"  test rows={len(te)}  ROC AUC={roc_auc_score(y[te], proba):.4f}  "
          f"Brier={brier_score_loss(y[te], proba):.5f} (AUC is circular - see header)")
    print("  calibration (predicted vs actual cheater rate by bin):")
    for lo, hi in [(0.0, 0.01), (0.01, 0.10), (0.10, 0.50), (0.50, 1.01)]:
        mask = (proba >= lo) & (proba < hi)
        if mask.sum():
            print(f"    pred {lo:.2f}-{hi:.2f}: n={int(mask.sum()):>6}  "
                  f"mean_pred={proba[mask].mean():.3f}  actual={y[te][mask].mean():.3f}")
    model_path = f"{out}/{VERSION}_cheater_scorer.pkl"
    joblib.dump(clf, model_path)

    def _sha(path):
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()

    return {"name": "cheater_scorer", "features": feats,
            "model_file": f"{VERSION}_cheater_scorer.pkl",
            "roc_auc": float(roc_auc_score(y[te], proba)),
            "model_sha256": _sha(model_path)}


def main(data=None, ablate=None):
    """Entry point. Works from the CLI and from a notebook.

    Jupyter/Colab inject their own argv (`-f /root/.../kernel-xxxx.json`), which
    argparse rejects with SystemExit(2). parse_known_args ignores those, and the
    keyword arguments let a notebook bypass argv entirely:
        main(data="user_backfilled.csv", ablate=True)
    """
    ap_ = argparse.ArgumentParser()
    # Paths are relative to the project root - run as `python train/v8_train.py`
    # (OUT_DIR="models" and the dataset both resolve from there, matching how
    # the root-level detectors are invoked).
    ap_.add_argument("--data", default="train/isac-ml-training-dataset.csv")
    ap_.add_argument("--ablate", action="store_true",
                     help="measure whether the criticalHits features help, across many seeds")
    args, unknown = ap_.parse_known_args()
    if unknown:
        print(f"[INFO] ignoring unrecognised args (notebook kernel?): {unknown}")
    # explicit keyword arguments win over argv
    if data is not None:
        args.data = data
    if ablate is not None:
        args.ablate = ablate
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_and_prepare(args.data)
    print(f"[INFO] {len(df)} rows, {df.identifier.nunique()} players, "
          f"{int(df.isCheater.sum())} cheater rows ({df.isCheater.mean():.2%})")
    players = audit_labels(df)

    # ---- STAGE 1: the rule ----
    print(f"\n[STAGE 1] player-level hs/h max > {AUTO_BAN_THRESHOLD} (auto)")
    m = players.hsmax > AUTO_BAN_THRESHOLD
    tp_ = int((m & (players.cheat==1)).sum()); fp_ = int((m & (players.cheat==0)).sum())
    fn_ = int((~m & (players.cheat==1)).sum())
    print(f"  flagged={tp_+fp_}  precision={tp_/(tp_+fp_):.4f}  recall={tp_/(tp_+fn_):.4f}")
    print(f"  {fp_} flagged-but-unlabelled players - review these, many are unlabelled cheaters")
    recall = tp_/(tp_+fn_)
    if recall > RECALL_CROSSOVER and DIVISOR_MODE == "tiered":
        print(f"  [WARN] recall {recall:.3f} is above the {RECALL_CROSSOVER:.2f} crossover.")
        print(f"         Past this point the tiered divisor ADDS false positives rather than")
        print(f"         removing them (+83% vs undamped at 90% recall). Either raise")
        print(f"         AUTO_BAN_THRESHOLD to land near {RECALL_CROSSOVER:.0%}, or set DIVISOR_MODE='smooth'.")

    band = (players.hsmax > REVIEW_BAND_FLOOR) & (players.hsmax <= AUTO_BAN_THRESHOLD)
    print(f"\n[BAND] {REVIEW_BAND_FLOOR:.0f}-{AUTO_BAN_THRESHOLD:.0f}: {int(band.sum())} players "
          f"({int((band & (players.cheat==1)).sum())} labelled cheaters) -> review, not auto-ban")

    # Stage 2 trains below the LABELLING threshold, not the (higher) auto-ban
    # line. The 805-1000 band is scored by this model at inference but is
    # excluded from training - its labels are the old rule's output.
    circ = set(players.index[players.hsmax > LABEL_RULE_THRESHOLD])
    resid = df[~df.identifier.isin(circ)].copy()
    print(f"[INFO] stage 2 training population: {resid.identifier.nunique()} players "
          f"below {LABEL_RULE_THRESHOLD:.0f} (non-circular labels only)")

    # ---- STAGE 2: a single review-queue model over the below-threshold pop ----
    results = []
    r = fit_stage2(resid, LIFETIME_FEATS, "stage2", OUT_DIR)
    if r: results.append(r)

    if args.ablate:
        crit_ablation(resid)

    # ---- Whole-dataset cheater_percentage model (display only, see header) ----
    scorer = fit_cheater_scorer(df, LIFETIME_FEATS, OUT_DIR)

    # ---- Queue bands, model-free reference ----
    print(f"\n[QUEUE] flat-threshold bands below the ban line")
    q = players[players.hsmax <= AUTO_BAN_THRESHOLD]
    for lo, hi in [(REVIEW_BAND_FLOOR, AUTO_BAN_THRESHOLD), (600, REVIEW_BAND_FLOOR), (500, 600), (0, 500)]:
        b = (q.hsmax > lo) & (q.hsmax <= hi)
        n = int(b.sum()); c = int((b & (q.cheat==1)).sum())
        if n:
            print(f"  {lo:>5.0f}-{hi:<5.0f}: {n:>6} players, {c:>4} known cheaters ({c/n:>6.2%})")

    with open(f"{OUT_DIR}/{VERSION}_manifest.json", "w") as f:
        import sklearn
        json.dump({"ban_threshold": AUTO_BAN_THRESHOLD,   # back-compat alias
                   "auto_ban_threshold": AUTO_BAN_THRESHOLD,
                   "review_band_floor": REVIEW_BAND_FLOOR,
                   "label_rule_threshold": LABEL_RULE_THRESHOLD,
                   "sklearn_version": sklearn.__version__, "queue_floor": QUEUE_FLOOR,
                   "inflation_factor": INFLATION_FACTOR, "seed": SEED,
                   "lifetime_features": LIFETIME_FEATS,
                   "crit_features": CRIT_FEATS,
                   "invariant_features": INVARIANT_FEATS,
                   "report_only_features": REPORT_ONLY_FEATS,
                   "cheater_scorer": scorer,
                   "stage2_models": results}, f, indent=2)
    print(f"\n[INFO] manifest -> {OUT_DIR}/{VERSION}_manifest.json")


if __name__ == "__main__":
    main()

# In a notebook, call main() directly instead of relying on argv:
#     main(data="user_backfilled.csv")
#     main(data="user_backfilled.csv", ablate=True)