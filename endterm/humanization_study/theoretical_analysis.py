"""
Deep theoretical analysis: Why R(s) fails and what would fix it.

R(s) = sum_ppl(s) / (2 * x_ppl(s))

sum_ppl = ppl(s, performer) + ppl(s_ideal, performer)
  - measures how far text is from greedy ideal
x_ppl = CE(softmax(observer), performer_logits)
  - measures disagreement between observer and performer

WHY HUMANIZATION BREAKS R(s):
- Humanization replaces high-probability tokens with lower-probability ones
- This increases ppl(s, performer) → sum_ppl goes up
- But x_ppl also changes because observer and performer NOW AGREE LESS
  about these unusual tokens → x_ppl goes up too
- The NET EFFECT: R(s) = sum_ppl / x_ppl increases, pushing toward "human"

THE FUNDAMENTAL PROBLEM:
R(s) captures ONE signal: "does this text follow the model's probability distribution?"
Human text: no → high R(s)
AI text: yes → low R(s)
Humanized AI text: no (deliberately) → high R(s) ← INDISTINGUISHABLE

WHAT WOULD A BETTER FUNCTION NEED?
It needs to capture a signal that:
1. Is HIGH for human text
2. Is LOW for AI text
3. Is ALSO LOW (or at least different) for humanized AI text

What property does humanized text have that human text doesn't?
- Humanized text is a TRANSFORMATION of AI text
- The mutations are SYSTEMATIC (all follow the same pattern)
- Human mutations are NATURAL (idiosyncratic, context-dependent)

POSSIBLE SIGNALS:
A. Mutation PATTERN: Human mutations cluster near semantic boundaries
   (creative word choice at key moments). AI humanization mutations are
   uniformly distributed (every banned word gets replaced).

B. Local vs Global consistency: Human text has consistent style throughout.
   Humanized text has style breaks (some parts untouched, some heavily rewritten).

C. Per-token perplexity VARIANCE: Human text has naturally varied perplexity.
   AI text has uniformly low perplexity. Humanized text has uniformly HIGH
   perplexity (because every replacement increases it equally).

   → Human: high variance in per-token perplexity
   → AI: low variance (uniformly predictable)
   → Humanized: low variance (uniformly unpredictable)
   → Both AI and humanized have LOW VARIANCE but different MEANS

D. Entropy of the perplexity distribution itself:
   Human: diverse (some tokens predictable, some surprising) → high entropy
   AI: concentrated at low perplexity → low entropy
   Humanized: concentrated at high perplexity → low entropy
   → Entropy of per-token perplexity could separate all three!

Let me test signal D with our existing data.
"""
import json, re, numpy as np
from collections import Counter
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.stats import entropy as scipy_entropy

BASE = Path(__file__).resolve().parent
RESULTS = BASE.parent.parent / "results"

falcon = json.load(open(RESULTS / "multimodel_results.json"))
all_scores = falcon["all_scores"]
dataset = json.load(open(BASE / "dataset" / "multimodel_p5e_dataset.json"))

human_texts = dataset["human_texts"]
ai_texts = dataset["original_ai_texts"]
humanized = dataset["humanized"]

# We need per-token perplexity variance, but we only have aggregate R(s).
# However, we CAN compute WORD-LEVEL features as proxy.
def words(text): return re.findall(r'\b[a-z]+\b', text.lower())

def word_frequency_entropy(text):
    """Entropy of word frequency distribution.
    Human: diverse word usage → higher entropy
    AI: repetitive patterns → lower entropy
    Humanized: forced diversity → potentially different pattern"""
    w = words(text)
    if len(w) < 5: return 0
    freq = Counter(w)
    probs = np.array(list(freq.values()), dtype=float)
    probs = probs / probs.sum()
    return float(scipy_entropy(probs))

def word_length_variance(text):
    """Variance of word lengths. Proxy for vocabulary diversity pattern."""
    w = words(text)
    if len(w) < 5: return 0
    lengths = [len(x) for x in w]
    return float(np.var(lengths))

def bigram_entropy(text):
    """Entropy of word bigrams. Captures structural diversity."""
    w = words(text)
    if len(w) < 5: return 0
    bigrams = [(w[i], w[i+1]) for i in range(len(w)-1)]
    freq = Counter(bigrams)
    probs = np.array(list(freq.values()), dtype=float)
    probs = probs / probs.sum()
    return float(scipy_entropy(probs))

def hapax_ratio(text):
    w = words(text); freq = Counter(w)
    return sum(1 for c in freq.values() if c == 1) / max(len(w), 1)

# Compute features for all groups
groups = {
    "human": human_texts,
    "original_ai": ai_texts,
    "kimi_p5e": humanized["moonshotai_kimi-k2-instruct"],
    "gemini_p5e": humanized["gemini-2.0-flash"],
}

FEATURES = {
    "word_freq_entropy": word_frequency_entropy,
    "word_length_var": word_length_variance,
    "bigram_entropy": bigram_entropy,
    "hapax_ratio": hapax_ratio,
}

print(f"{'Feature':<22}", end="")
for g in groups: print(f" {g:>12}", end="")
print()
print("="*70)

for fname, ffn in FEATURES.items():
    print(f"{fname:<22}", end="")
    for gname, texts in groups.items():
        vals = [ffn(t) for t in texts if t]
        print(f" {np.mean(vals):>12.4f}", end="")
    print()

# Test discriminative power of each feature
print(f"\n{'Feature':<22} {'H vs AI':>8} {'H vs Kimi':>10} {'H vs Gem':>9}")
print("-"*52)
for fname, ffn in FEATURES.items():
    hv = [ffn(t) for t in human_texts if t]
    av = [ffn(t) for t in ai_texts if t]
    kv = [ffn(t) for t in humanized["moonshotai_kimi-k2-instruct"] if t]
    gv = [ffn(t) for t in humanized["gemini-2.0-flash"] if t]

    # AUROC: higher feature = more human
    a_ai = roc_auc_score([1]*len(hv)+[0]*len(av), hv+av) if len(av)>0 else 0
    a_k = roc_auc_score([1]*len(hv)+[0]*len(kv), hv+kv) if len(kv)>0 else 0
    a_g = roc_auc_score([1]*len(hv)+[0]*len(gv), hv+gv) if len(gv)>0 else 0
    print(f"{fname:<22} {a_ai:>8.4f} {a_k:>10.4f} {a_g:>9.4f}")

# Now try: R(s) combined with the best discriminative feature
print("\n=== COMBINED SCORES: R(s) * feature_signal ===")
print("Goal: find a combination that is HIGH for human, LOW for both AI AND humanized")

# The key insight: we need a function where
# human → high, AI → low, humanized → low
# R(s): human=high, AI=low, humanized=HIGH (fails)
# We need something that's LOW for humanized but HIGH for human

# What if we multiply R(s) by a "naturalness" score?
# Naturalness: how NORMAL is the text's diversity compared to human baseline?
# Not just "is it too diverse" but "is the PATTERN of diversity natural?"

# Proposal: Use |feature - human_mean| as a deviation score
# Both AI (too low diversity) and humanized (too high diversity) deviate from human

for fname, ffn in FEATURES.items():
    hv_feat = [ffn(t) for t in human_texts if t]
    f_mean, f_std = np.mean(hv_feat), np.std(hv_feat)

    def naturalness_score(rs, text, f_mean=f_mean, f_std=f_std, ffn=ffn):
        """Penalize deviation from human mean in EITHER direction"""
        feat = ffn(text)
        z_score = abs(feat - f_mean) / max(f_std, 0.001)
        penalty = np.exp(-0.5 * max(0, z_score - 1.0))  # Gaussian penalty beyond 1 std
        return rs * penalty

    # Evaluate
    h_s = []; ai_s = []; k_s = []; g_s = []
    for i in range(len(human_texts)):
        r = all_scores["human"][i]
        if r and r == r and human_texts[i]:
            h_s.append(naturalness_score(r, human_texts[i]))
    for i in range(len(ai_texts)):
        r = all_scores["original_ai"][i]
        if r and r == r and ai_texts[i]:
            ai_s.append(naturalness_score(r, ai_texts[i]))
    kimi_texts = humanized["moonshotai_kimi-k2-instruct"]
    kimi_scores = all_scores.get("moonshotai_kimi-k2-instruct", [])
    for i in range(min(len(kimi_texts), len(kimi_scores))):
        r = kimi_scores[i]
        if r and r == r and kimi_texts[i]:
            k_s.append(naturalness_score(r, kimi_texts[i]))
    gem_texts = humanized["gemini-2.0-flash"]
    gem_scores = all_scores.get("gemini-2.0-flash", [])
    for i in range(min(len(gem_texts), len(gem_scores))):
        r = gem_scores[i]
        if r and r == r and gem_texts[i]:
            g_s.append(naturalness_score(r, gem_texts[i]))

    if h_s and ai_s and k_s:
        a_ai = roc_auc_score([1]*len(h_s)+[0]*len(ai_s), h_s+ai_s)
        a_k = roc_auc_score([1]*len(h_s)+[0]*len(k_s), h_s+k_s)
        a_g = roc_auc_score([1]*len(h_s)+[0]*len(g_s), h_s+g_s) if g_s else 0
        print(f"R(s)*nat({fname[:12]:12}) vs_AI={a_ai:.4f} vs_Kimi={a_k:.4f} vs_Gem={a_g:.4f}")

print("\n=== REFERENCE: R(s) alone ===")
h_r = [s for s in all_scores["human"] if s and s==s]
ai_r = [s for s in all_scores["original_ai"] if s and s==s]
k_r = [s for s in all_scores.get("moonshotai_kimi-k2-instruct",[]) if s and s==s]
g_r = [s for s in all_scores.get("gemini-2.0-flash",[]) if s and s==s]
print(f"R(s) alone:           vs_AI={roc_auc_score([1]*len(h_r)+[0]*len(ai_r), h_r+ai_r):.4f} "
      f"vs_Kimi={roc_auc_score([1]*len(h_r)+[0]*len(k_r), h_r+k_r):.4f} "
      f"vs_Gem={roc_auc_score([1]*len(h_r)+[0]*len(g_r), h_r+g_r):.4f}")
