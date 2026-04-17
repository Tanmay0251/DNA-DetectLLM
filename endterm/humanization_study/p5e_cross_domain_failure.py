"""
P5E Cross-Domain Failure Analysis
==================================
Show that P5E (derived from XSum) bans words that are NORMAL in other domains.
This proves P5E is domain-specific and motivates the literature-based approach.
"""
import json, re
from collections import Counter
from pathlib import Path

DATA = Path("/mnt/c/Users/manda/Desktop/Sem8/IE663/DNA-DetectLLM-repo/Data")

P5E_BANNED = set("crucial comprehensive testament unwavering numerous determination importance "
    "unique potentially various maintain maintaining additionally resilience ensuring "
    "dedication strategic amidst resulted highlight atmosphere multiple sustainable "
    "broader valuable emotional beloved perspective emphasized implement offerings "
    "stance outcomes furthermore moreover pivotal landscape foster underscore paramount".split())

def words(text):
    return re.findall(r'\b[a-z]+\b', text.lower())

def banned_in_text(text):
    w = words(text)
    found = [word for word in w if word in P5E_BANNED]
    return found

# Load human texts from each domain
domains = {
    "XSum (news)": json.load(open(DATA / "Collected data" / "xsum_human.json"))["human_text"][:200],
    "WritingPrompts (creative)": json.load(open(DATA / "Collected data" / "wp_human.json"))["human_text"][:200],
    "ArXiv (academic)": json.load(open(DATA / "Collected data" / "arxiv_human.json"))["human_text"][:200],
    "M4 (multi-domain)": json.load(open(DATA / "M4" / "M4_human_test.json")).get("human_text", json.load(open(DATA / "M4" / "M4_human_test.json")))[:200] if isinstance(json.load(open(DATA / "M4" / "M4_human_test.json")), dict) else json.load(open(DATA / "M4" / "M4_human_test.json"))[:200],
}

# Also load GPT-4 AI texts
gpt4 = json.load(open(DATA / "Collected data" / "GPT4_machine_test.json"))["machine_text"][:200]

print("="*70)
print("P5E CROSS-DOMAIN ANALYSIS: Banned words in human text by domain")
print("="*70)
print(f"\nP5E bans {len(P5E_BANNED)} words. If human text in a domain uses these words,")
print("P5E humanization would REMOVE natural human vocabulary = unfair advantage.\n")

print(f"{'Domain':<30} {'Texts':>6} {'With banned':>12} {'%':>6} {'Avg count':>10}")
print("-"*70)

for domain, texts in domains.items():
    has_banned = 0
    total_banned = 0
    for t in texts:
        if isinstance(t, str):
            found = banned_in_text(t)
            if found:
                has_banned += 1
            total_banned += len(found)
    pct = has_banned / len(texts) * 100
    avg = total_banned / len(texts)
    print(f"{domain:<30} {len(texts):>6} {has_banned:>12} {pct:>5.1f}% {avg:>10.2f}")

# GPT-4 for comparison
has_b = sum(1 for t in gpt4 if banned_in_text(t))
avg_b = sum(len(banned_in_text(t)) for t in gpt4) / len(gpt4)
print(f"{'GPT-4 AI (for comparison)':<30} {len(gpt4):>6} {has_b:>12} {has_b/len(gpt4)*100:>5.1f}% {avg_b:>10.2f}")

# Show which P5E words appear in ArXiv human text
print(f"\n{'='*70}")
print("CRITICAL: P5E banned words that appear in ArXiv HUMAN text")
print("(banning these removes natural academic vocabulary)")
print(f"{'='*70}")

arxiv_texts = domains["ArXiv (academic)"]
arxiv_counter = Counter()
for t in arxiv_texts:
    if isinstance(t, str):
        arxiv_counter.update(words(t))

print(f"\n{'Banned word':<20} {'ArXiv human freq':>16} {'XSum human freq':>15} {'Problem?':>10}")
print("-"*65)

xsum_counter = Counter()
for t in domains["XSum (news)"]:
    if isinstance(t, str):
        xsum_counter.update(words(t))

problems = 0
for word in sorted(P5E_BANNED):
    af = arxiv_counter.get(word, 0)
    xf = xsum_counter.get(word, 0)
    problem = "YES" if af >= 3 else ""
    if problem:
        problems += 1
    print(f"{word:<20} {af:>16} {xf:>15} {problem:>10}")

print(f"\n{problems}/{len(P5E_BANNED)} P5E banned words appear ≥3 times in ArXiv human text.")
print("\nCONCLUSION: P5E is XSum-specific. On ArXiv, it would ban words that")
print("real academics use, making the test unfair. A domain-agnostic prompt")
print("(based on published AI writing research) is needed for honest evaluation.")
