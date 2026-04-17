"""
Improved Score Function Analysis
=================================
DNA-DetectLLM's R(s) = sum_ppl / (2 * x_ppl) fails against humanization.
Can we augment it with cheap text features to recover detection?

Key insight from our experiments:
- P5E humanized text has high R(s) (fools detector) BUT also has
  unnaturally high lexical diversity (hapax ratio, type-token ratio)
- This is because the humanization prompt FORCES diverse word replacements
- Real human text has NATURAL word reuse patterns

Proposed: R_aug(s) = R(s) × diversity_penalty(s)
Where diversity_penalty penalizes texts with lexical diversity
significantly higher than the human baseline.

This is a GENERAL defense — it doesn't depend on knowing the specific
humanization prompt or banned word list. It catches the SIDE EFFECT
of any vocabulary-diversifying attack.
"""
import json, re, numpy as np
from collections import Counter
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve

BASE = Path(__file__).resolve().parent
RESULTS = BASE.parent.parent / "results"

# Load Falcon scores and texts
falcon = json.load(open(RESULTS / "multimodel_results.json"))
all_scores = falcon["all_scores"]

dataset = json.load(open(BASE / "dataset" / "multimodel_p5e_dataset.json"))
human_texts = dataset["human_texts"]
ai_texts = dataset["original_ai_texts"]
humanized = dataset["humanized"]

# ── Text features ───────────────────────────────────────────────
def words(text):
    return re.findall(r'\b[a-z]+\b', text.lower())

def hapax_ratio(text):
    w = words(text)
    freq = Counter(w)
    return sum(1 for c in freq.values() if c == 1) / max(len(w), 1)

def type_token_ratio(text):
    w = words(text)
    return len(set(w)) / max(len(w), 1)

# ── Compute human baseline stats ───────────────────────────────
h_hapax = [hapax_ratio(t) for t in human_texts if t]
h_ttr = [type_token_ratio(t) for t in human_texts if t]
HAPAX_MEAN = np.mean(h_hapax)
HAPAX_STD = np.std(h_hapax)
TTR_MEAN = np.mean(h_ttr)
TTR_STD = np.std(h_ttr)

print(f"Human baseline: hapax={HAPAX_MEAN:.4f}±{HAPAX_STD:.4f}, TTR={TTR_MEAN:.4f}±{TTR_STD:.4f}")

# ── Score functions to test ─────────────────────────────────────
def R_original(r_s, text):
    return r_s

def R_hapax(r_s, text, k=1.0):
    h = hapax_ratio(text)
    excess = max(0, h - (HAPAX_MEAN + k * HAPAX_STD))
    return r_s * max(0.1, 1 - 2.0 * excess)

def R_ttr(r_s, text, k=1.0):
    t = type_token_ratio(text)
    excess = max(0, t - (TTR_MEAN + k * TTR_STD))
    return r_s * max(0.1, 1 - 2.0 * excess)

def R_combined(r_s, text, k=1.0, alpha=1.5, beta=1.5):
    h = hapax_ratio(text)
    t = type_token_ratio(text)
    h_excess = max(0, h - (HAPAX_MEAN + k * HAPAX_STD))
    t_excess = max(0, t - (TTR_MEAN + k * TTR_STD))
    penalty = 1 - alpha * h_excess - beta * t_excess
    return r_s * max(0.1, penalty)

# ── Evaluate all score functions ────────────────────────────────
SCORERS = {
    "R(s) original": R_original,
    "R_hapax (k=1.0)": lambda r, t: R_hapax(r, t, k=1.0),
    "R_hapax (k=1.5)": lambda r, t: R_hapax(r, t, k=1.5),
    "R_ttr (k=1.0)": lambda r, t: R_ttr(r, t, k=1.0),
    "R_combined (k=0.5)": lambda r, t: R_combined(r, t, k=0.5),
    "R_combined (k=1.0)": lambda r, t: R_combined(r, t, k=1.0),
    "R_combined (k=1.5)": lambda r, t: R_combined(r, t, k=1.5),
}

groups = {
    "Kimi-K2": "moonshotai_kimi-k2-instruct",
    "Gemini": "gemini-2.0-flash",
    "Qwen": "qwen_qwen3-32b",
    "Llama-70B": "llama-3.3-70b-versatile",
    "Original AI": "original_ai",
}

print(f"\n{'Score Function':<25}", end="")
for label in groups:
    print(f" {label:>12}", end="")
print()
print("="*90)

for scorer_name, scorer_fn in SCORERS.items():
    print(f"{scorer_name:<25}", end="")

    # Human scores
    h_new = []
    for i in range(len(human_texts)):
        r = all_scores["human"][i]
        if r is not None and r == r and human_texts[i]:
            h_new.append(scorer_fn(r, human_texts[i]))

    for label, mk in groups.items():
        texts = ai_texts if mk == "original_ai" else humanized.get(mk, [])
        scores_list = all_scores.get(mk, [])
        g_new = []
        for i in range(min(len(texts), len(scores_list))):
            r = scores_list[i]
            if r is not None and r == r and i < len(texts) and texts[i]:
                g_new.append(scorer_fn(r, texts[i]))

        if len(g_new) > 0 and len(h_new) > 0:
            auroc = roc_auc_score([1]*len(h_new) + [0]*len(g_new), h_new + g_new)
            print(f" {auroc:>12.4f}", end="")
        else:
            print(f" {'N/A':>12}", end="")
    print()

print("\n" + "="*90)
print("INTERPRETATION:")
print("- R(s) original: Kimi AUROC=0.197 (detector broken)")
print("- R_combined: Kimi AUROC should increase significantly")
print("- Key: the penalty should NOT hurt Original AI detection too much")
print("- Sweet spot: k=1.0 or k=1.5 (how many std devs above human mean to allow)")
