"""Run P5E humanization for a single model, save to per-model file."""
import json, time, os, sys, re, requests
from pathlib import Path

GROQ_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY")
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_TEXTS = json.load(open(SCRIPT_DIR / "dataset" / "gpt4_base_texts.json"))[:100]

P5E_PROMPT = (
    "Rewrite this text while strictly avoiding these words and patterns:\n\n"
    "BANNED WORDS: crucial, comprehensive, testament, unwavering, numerous, "
    "determination, importance, unique, potentially, various, maintain, maintaining, "
    "additionally, resilience, ensuring, dedication, strategic, amidst, resulted, "
    "highlight, atmosphere, multiple, sustainable, broader, valuable, emotional, "
    "beloved, perspective, emphasized, implement, offerings, stance, outcomes, "
    "furthermore, moreover, pivotal, landscape, foster, underscore, paramount.\n\n"
    "BANNED PATTERNS: Do not use parallel sentence structures. Do not start "
    "paragraphs with a topic sentence followed by supporting details. "
    "Avoid lists of three. Never use 'It is important to note' or "
    "'It is worth mentioning' or 'In conclusion' or 'As a result'.\n\n"
    "Replace any banned word with a SPECIFIC, CONCRETE alternative — "
    "not a generic synonym. Same meaning, same approximate length. "
    "Output only the rewritten text.\n\n"
    "Text to rewrite:\n{text}"
)

def strip_thinking(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def call_api(model_id, prompt):
    is_thinking = "qwen" in model_id.lower()
    messages = [{"role": "user", "content": prompt}]
    if is_thinking:
        messages.insert(0, {"role": "system", "content": "Respond directly without thinking. Do not use <think> tags. Output only the rewritten text."})

    for attempt in range(3):
        try:
            resp = requests.post("https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
                json={"model": model_id, "messages": messages, "temperature": 0.7, "max_tokens": 2048},
                timeout=90)
            if resp.status_code == 429:
                wait = min(60, 10 * (2 ** attempt))
                print(f" [wait {wait}s]", end="", flush=True)
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                time.sleep(5)
                continue
            text = strip_thinking(resp.json()["choices"][0]["message"]["content"].strip())
            if len(text) > 30:
                return text
        except Exception as e:
            time.sleep(5)
    return None

def main():
    model_id = sys.argv[1]
    safe_name = model_id.replace("/", "_")
    out_file = SCRIPT_DIR / "dataset" / f"model_{safe_name}.json"

    # Load existing or create new
    results = [None] * 100
    if out_file.exists():
        results = json.load(open(out_file))
        done = sum(1 for x in results if x is not None)
        print(f"Resuming {model_id}: {done}/100 done")

    done = sum(1 for x in results if x is not None)
    print(f"\n=== {model_id}: {done}/100, {100-done} remaining ===")

    for i in range(100):
        if results[i] is not None:
            continue
        prompt = P5E_PROMPT.format(text=BASE_TEXTS[i])
        print(f"  [{i+1}/100]", end="", flush=True)
        result = call_api(model_id, prompt)
        if result:
            results[i] = result
            print(f" OK ({len(result)})")
        else:
            print(f" FAIL")
        time.sleep(4)
        if (i+1) % 10 == 0:
            json.dump(results, open(out_file, "w"))
            print(f"  [saved]")

    json.dump(results, open(out_file, "w"))
    done = sum(1 for x in results if x is not None)
    print(f"\nDone: {done}/100 saved to {out_file}")

if __name__ == "__main__":
    main()
