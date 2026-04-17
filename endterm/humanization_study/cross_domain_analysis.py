"""
Cross-Domain & Cross-Model Word Frequency Analysis
===================================================
Goal: Find AI-overrepresented words that are universal (not domain-specific).

Approach:
1. For each domain (XSum, WritingPrompts, ArXiv): count AI vs Human word freqs
2. For each AI model (GPT-4, Claude, Gemini): count word freqs
3. Find words overrepresented ACROSS ALL domains and models
   → These are instruction-tuning artifacts, not domain artifacts
4. Build a domain-agnostic humanization prompt
"""
import json, re
from collections import Counter
from pathlib import Path

DATA = Path("/mnt/c/Users/manda/Desktop/Sem8/IE663/DNA-DetectLLM-repo/Data/Collected data")

# Load all datasets
datasets = {
    "xsum_human": json.load(open(DATA / "xsum_human.json"))["human_text"][:200],
    "wp_human": json.load(open(DATA / "wp_human.json"))["human_text"][:200],
    "arxiv_human": json.load(open(DATA / "arxiv_human.json"))["human_text"][:200],
    "gpt4_ai": json.load(open(DATA / "GPT4_machine_test.json"))["machine_text"][:200],
    "claude_ai": json.load(open(DATA / "Claude_machine_test.json"))["machine_text"][:200],
    "gemini_ai": json.load(open(DATA / "Gemini_machine_test.json"))["machine_text"][:200],
}

for name, texts in datasets.items():
    print(f"{name}: {len(texts)} texts")

def words(text):
    return re.findall(r'\b[a-z]+\b', text.lower())

STOPWORDS = set("the and for that this with from have been will were their also than more over into "
    "such most some when what only made after year many them time they each work part well both come "
    "like make just very take much back long even good give area used while those other being about "
    "would could which there where should these since through between during before across around among "
    "said told asked added noted stated described according called known found reported showed "
    "first last next then here there still just does done going than been".split())

def count_words(texts):
    c = Counter()
    for t in texts:
        c.update(w for w in words(t) if len(w) >= 4 and w not in STOPWORDS)
    return c

# Count per dataset
counters = {name: count_words(texts) for name, texts in datasets.items()}

# ── Per-domain overrepresentation ───────────────────────────────
print("\n" + "="*80)
print("PER-DOMAIN: AI-overrepresented words (AI freq >= 5, Human freq <= 1)")
print("="*80)

domain_pairs = [
    ("gpt4_ai", "xsum_human", "GPT-4 vs XSum"),
    ("gpt4_ai", "wp_human", "GPT-4 vs WritingPrompts"),
    ("gpt4_ai", "arxiv_human", "GPT-4 vs ArXiv"),
]

per_domain_overrep = {}
for ai_key, h_key, label in domain_pairs:
    ai_c = counters[ai_key]
    h_c = counters[h_key]
    overrep = set()
    for word, freq in ai_c.items():
        if freq >= 5 and h_c.get(word, 0) <= 1:
            overrep.add(word)
    per_domain_overrep[label] = overrep
    print(f"\n{label}: {len(overrep)} overrepresented words")

# Find INTERSECTION — words overrepresented in ALL domains
all_domains = list(per_domain_overrep.values())
universal_words = all_domains[0]
for s in all_domains[1:]:
    universal_words = universal_words & s

print(f"\n{'='*80}")
print(f"UNIVERSAL: Overrepresented in ALL 3 domains: {len(universal_words)} words")
print(f"{'='*80}")

# Sort by GPT-4 frequency
gpt4_c = counters["gpt4_ai"]
universal_sorted = sorted(universal_words, key=lambda w: gpt4_c[w], reverse=True)

print(f"\n{'Word':<20} {'GPT4':>6} {'XSum_H':>7} {'WP_H':>7} {'ArX_H':>7}")
print("-"*50)
for w in universal_sorted[:50]:
    print(f"{w:<20} {gpt4_c[w]:>6} {counters['xsum_human'].get(w,0):>7} "
          f"{counters['wp_human'].get(w,0):>7} {counters['arxiv_human'].get(w,0):>7}")

# ── Per-model analysis ──────────────────────────────────────────
print(f"\n{'='*80}")
print("PER-MODEL: Words overrepresented by each AI model vs ALL human domains")
print(f"{'='*80}")

# Combine all human texts
all_human = Counter()
for key in ["xsum_human", "wp_human", "arxiv_human"]:
    all_human += counters[key]

model_overrep = {}
for model_key, model_name in [("gpt4_ai", "GPT-4"), ("claude_ai", "Claude"), ("gemini_ai", "Gemini")]:
    mc = counters[model_key]
    overrep = set()
    for word, freq in mc.items():
        # AI freq >= 5 AND human freq <= 2 (across all 600 human texts)
        if freq >= 5 and all_human.get(word, 0) <= 2:
            overrep.add(word)
    model_overrep[model_name] = overrep
    print(f"\n{model_name}: {len(overrep)} overrepresented words (vs all human)")

# Words shared by ALL models
shared_all = model_overrep["GPT-4"] & model_overrep["Claude"] & model_overrep["Gemini"]
print(f"\nShared by ALL 3 models: {len(shared_all)} words")
print("These are INSTRUCTION-TUNING ARTIFACTS (not model-specific):")
shared_sorted = sorted(shared_all, key=lambda w: gpt4_c[w] + counters["claude_ai"][w] + counters["gemini_ai"][w], reverse=True)
for w in shared_sorted[:40]:
    print(f"  {w:<20} GPT4={gpt4_c[w]:>3} Claude={counters['claude_ai'][w]:>3} Gemini={counters['gemini_ai'][w]:>3} Human={all_human.get(w,0):>3}")

# ── Check if P5E words are universal or domain-specific ─────────
P5E = "crucial comprehensive testament unwavering numerous determination importance unique potentially various maintain maintaining additionally resilience ensuring dedication strategic amidst resulted highlight atmosphere multiple sustainable broader valuable emotional beloved perspective emphasized implement offerings stance outcomes furthermore moreover pivotal landscape foster underscore paramount".split()

print(f"\n{'='*80}")
print("P5E VALIDATION: Are P5E words universal or XSum-specific?")
print(f"{'='*80}")
p5e_universal = 0
p5e_xsum_only = 0
for w in P5E:
    in_universal = w in universal_words
    in_shared = w in shared_all
    gf = gpt4_c.get(w, 0)
    xf = counters["xsum_human"].get(w, 0)
    wf = counters["wp_human"].get(w, 0)
    af = counters["arxiv_human"].get(w, 0)
    status = "UNIVERSAL" if in_universal else ("SHARED-MODEL" if in_shared else "XSum-SPECIFIC")
    if in_universal:
        p5e_universal += 1
    elif not in_shared:
        p5e_xsum_only += 1
    print(f"  {w:<20} {status:<15} GPT4={gf:>2} XSum_H={xf:>2} WP_H={wf:>2} ArX_H={af:>2}")

print(f"\n{p5e_universal}/{len(P5E)} P5E words are domain-universal")
print(f"{p5e_xsum_only}/{len(P5E)} P5E words may be XSum-specific")

# ── Build domain-agnostic prompt ────────────────────────────────
print(f"\n{'='*80}")
print("PROPOSED DOMAIN-AGNOSTIC BANNED WORD LIST")
print("(overrepresented by ALL AI models across ALL domains)")
print(f"{'='*80}")
# Use words shared by at least 2 models AND universal across domains
robust_words = set()
for w in universal_words:
    model_count = sum(1 for m in model_overrep.values() if w in m)
    if model_count >= 2:
        robust_words.add(w)

# Also add shared_all words even if not in all domains
for w in shared_all:
    robust_words.add(w)

print(f"\nRobust banned words: {len(robust_words)}")
robust_sorted = sorted(robust_words, key=lambda w: gpt4_c.get(w,0), reverse=True)
print(", ".join(robust_sorted[:50]))
