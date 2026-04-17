"""
Word Frequency Analysis: How the P5E banned word list was derived.
================================================================
This is the data-driven methodology behind the P5E prompt.

Step 1: Count word frequencies in 100 GPT-4 texts and 100 XSum human texts
Step 2: Find words that appear 5+ times in AI text but 0-1 times in human
Step 3: These become the P5E banned word list

This script reproduces the analysis and shows the evidence.
"""
import json, re
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent / "dataset"

# Load the original 100 GPT-4 and 100 human texts
ai_texts = json.load(open(BASE / "gpt4_base_texts.json"))[:100]
dataset = json.load(open(BASE / "p5e_dataset.json"))
human_texts = dataset["human_texts"][:100]

def words(text):
    return re.findall(r'\b[a-z]+\b', text.lower())

# Count word frequencies
ai_counter = Counter()
human_counter = Counter()

for t in ai_texts:
    ai_counter.update(words(t))
for t in human_texts:
    human_counter.update(words(t))

print(f"AI vocabulary: {len(ai_counter)} unique words, {sum(ai_counter.values())} total")
print(f"Human vocabulary: {len(human_counter)} unique words, {sum(human_counter.values())} total")

# Find overrepresented AI words: appears 5+ in AI, 0-1 in human
print("\n" + "="*70)
print("OVERREPRESENTED AI WORDS (AI freq >= 5, Human freq <= 1)")
print("="*70)
print(f"{'Word':<20} {'AI freq':>8} {'Human freq':>10} {'Ratio':>8}")
print("-"*50)

overrep = []
for word, ai_freq in ai_counter.most_common():
    human_freq = human_counter.get(word, 0)
    if ai_freq >= 5 and human_freq <= 1:
        # Skip common stopwords and very short words
        if len(word) >= 4 and word not in {'that', 'this', 'with', 'from', 'have', 'been',
            'will', 'were', 'their', 'also', 'than', 'more', 'over', 'into', 'such',
            'most', 'some', 'when', 'what', 'only', 'made', 'after', 'year', 'many',
            'them', 'time', 'they', 'each', 'work', 'part', 'well', 'both', 'come',
            'like', 'make', 'just', 'very', 'take', 'much', 'back', 'long', 'even',
            'good', 'give', 'area', 'used', 'while', 'those', 'other', 'being', 'about',
            'would', 'could', 'which', 'there', 'where', 'should', 'these', 'since',
            'through', 'between', 'during', 'before', 'across', 'around', 'among'}:
            overrep.append((word, ai_freq, human_freq))
            print(f"{word:<20} {ai_freq:>8} {human_freq:>10} {ai_freq/max(human_freq,0.5):>8.1f}x")

print(f"\nTotal overrepresented words: {len(overrep)}")

# Now check the actual P5E banned list
P5E_BANNED = "crucial comprehensive testament unwavering numerous determination importance unique potentially various maintain maintaining additionally resilience ensuring dedication strategic amidst resulted highlight atmosphere multiple sustainable broader valuable emotional beloved perspective emphasized implement offerings stance outcomes furthermore moreover pivotal landscape foster underscore paramount".split()

print("\n" + "="*70)
print("P5E BANNED WORD VALIDATION")
print("="*70)
print(f"{'Banned word':<20} {'AI freq':>8} {'Human freq':>10} {'In overrep?':>12}")
print("-"*55)

for word in sorted(P5E_BANNED):
    ai_f = ai_counter.get(word, 0)
    h_f = human_counter.get(word, 0)
    in_list = "YES" if ai_f >= 5 and h_f <= 1 else "no"
    print(f"{word:<20} {ai_f:>8} {h_f:>10} {in_list:>12}")

# Also check P5 (original) words that DON'T appear in the data
P5_ORIGINAL = "delve crucial pivotal robust leverage facilitate utilize comprehensive furthermore moreover additionally realm tapestry landscape testament foster underscore multifaceted intricate nuanced noteworthy commendable meticulous paramount indispensable embark embrace".split()

print("\n" + "="*70)
print("P5 (ORIGINAL) vs DATA — why P5 was suboptimal")
print("="*70)
absent = []
for word in sorted(P5_ORIGINAL):
    ai_f = ai_counter.get(word, 0)
    h_f = human_counter.get(word, 0)
    status = "NEVER IN DATA" if ai_f == 0 else f"AI={ai_f}, H={h_f}"
    if ai_f == 0:
        absent.append(word)
    print(f"  {word:<20} {status}")

print(f"\n{len(absent)}/{len(P5_ORIGINAL)} P5 words NEVER appear in GPT-4 data:")
print(f"  {', '.join(absent)}")
print(f"\nThis is why P5E (data-driven) outperforms P5 (hand-picked).")
