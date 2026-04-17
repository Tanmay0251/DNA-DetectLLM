"""
Multi-Model P5E Humanization Generation
========================================
Takes 100 GPT-4 base texts, applies P5E prompt via multiple Groq models.
Gemini results already exist — this adds Llama, Qwen, Llama-4-Scout, GPT-OSS.

Usage:
    python3 generate_multimodel.py
    python3 generate_multimodel.py --models llama-3.3-70b-versatile qwen/qwen3-32b
    python3 generate_multimodel.py --resume
"""

import json, time, os, sys, argparse
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────
GROQ_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY")
N_TEXTS = 100
MAX_RETRIES = 3
CHECKPOINT_EVERY = 10

MODELS = {
    "llama-3.3-70b-versatile":                    "Llama-3.3-70B",
    "qwen/qwen3-32b":                             "Qwen-3-32B",
    "meta-llama/llama-4-scout-17b-16e-instruct":  "Llama-4-Scout-17B",
    "llama-3.1-8b-instant":                       "Llama-3.1-8B",
    "openai/gpt-oss-120b":                        "GPT-OSS-120B",
    "moonshotai/kimi-k2-instruct":                "Kimi-K2",
}

# Rate limits per model (requests per minute, be conservative)
RATE_LIMITS = {
    "llama-3.3-70b-versatile": 30,
    "qwen/qwen3-32b": 30,
    "meta-llama/llama-4-scout-17b-16e-instruct": 30,
    "llama-3.1-8b-instant": 30,
    "openai/gpt-oss-120b": 30,
    "moonshotai/kimi-k2-instruct": 30,
}

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

# ── Paths ───────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR / "dataset"
BASE_TEXTS_FILE = DATASET_DIR / "gpt4_base_texts.json"
OUTPUT_FILE = DATASET_DIR / "multimodel_p5e_dataset.json"
CHECKPOINT_FILE = DATASET_DIR / "multimodel_checkpoint.json"


def load_base_texts():
    with open(BASE_TEXTS_FILE) as f:
        texts = json.load(f)
    print(f"Loaded {len(texts)} base texts")
    return texts[:N_TEXTS]


def strip_thinking(text):
    """Strip <think>...</think> tags from thinking models like Qwen."""
    import re
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return text


def call_groq(model_id, prompt, retries=MAX_RETRIES):
    """Call Groq API with retries."""
    import requests
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"}

    # Thinking models need special handling
    is_thinking = "qwen" in model_id.lower()
    messages = [{"role": "user", "content": prompt}]
    if is_thinking:
        messages.insert(0, {"role": "system", "content": "Respond directly without thinking. Do not use <think> tags. Output only the rewritten text."})

    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, json={
                "model": model_id,
                "messages": messages,
                "temperature": 0.6 if is_thinking else 0.7,
                "max_tokens": 2048,
            }, timeout=90)

            if resp.status_code == 429:
                wait = min(60, 10 * (2 ** attempt))
                print(f" [rate-limit, wait {wait}s]", end="", flush=True)
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                print(f" [HTTP {resp.status_code}]", end="", flush=True)
                time.sleep(5)
                continue

            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()
            text = strip_thinking(text)
            if len(text) > 30:
                return text
            print(f" [short response: {len(text)} chars]", end="", flush=True)

        except Exception as e:
            wait = 3 * (2 ** attempt)
            print(f" [error: {e}, wait {wait}s]", end="", flush=True)
            time.sleep(wait)

    return None


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return None


def save_checkpoint(dataset):
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(dataset, f)
    print(f"  [checkpoint saved]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None, help="Specific model IDs to run")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    base_texts = load_base_texts()

    # Select models
    model_ids = args.models if args.models else list(MODELS.keys())

    # Load or init dataset
    dataset = None
    if args.resume:
        dataset = load_checkpoint()
        if dataset:
            print("Resumed from checkpoint")

    if not dataset:
        dataset = {
            "metadata": {
                "n_base_texts": N_TEXTS,
                "source_model": "GPT-4 Turbo",
                "prompt": "P5E_enhanced_lexicon",
                "prompt_text": P5E_PROMPT,
                "models": {k: v for k, v in MODELS.items() if k in model_ids},
            },
            "human_texts": [],  # will be filled from p5e_dataset.json
            "original_ai_texts": base_texts,
            "humanized": {},
        }

    # Load human texts from existing dataset
    p5e_file = DATASET_DIR / "p5e_dataset.json"
    if p5e_file.exists() and not dataset["human_texts"]:
        with open(p5e_file) as f:
            dataset["human_texts"] = json.load(f)["human_texts"][:N_TEXTS]
        print(f"Loaded {len(dataset['human_texts'])} human texts from p5e_dataset.json")

    # ── Generate ────────────────────────────────────────────────────
    for model_id in model_ids:
        label = MODELS.get(model_id, model_id)
        key = model_id.replace("/", "_")

        if key not in dataset["humanized"]:
            dataset["humanized"][key] = [None] * N_TEXTS

        done = sum(1 for x in dataset["humanized"][key] if x is not None)
        remaining = N_TEXTS - done

        if remaining == 0:
            print(f"\n[{label}] Already complete ({N_TEXTS}/{N_TEXTS})")
            continue

        rpm = RATE_LIMITS.get(model_id, 30)
        delay = 60.0 / rpm + 2.0  # generous safety margin

        print(f"\n{'='*60}")
        print(f"[{label}] ({model_id})")
        print(f"  {done}/{N_TEXTS} done, {remaining} remaining, {delay:.1f}s between calls")
        print(f"{'='*60}")

        for i in range(N_TEXTS):
            if dataset["humanized"][key][i] is not None:
                continue

            prompt = P5E_PROMPT.format(text=base_texts[i])
            print(f"  [{i+1}/{N_TEXTS}]", end="", flush=True)

            result = call_groq(model_id, prompt)
            if result:
                dataset["humanized"][key][i] = result
                print(f" OK ({len(result)} chars)")
            else:
                print(f" FAILED")

            time.sleep(delay)

            if (i + 1) % CHECKPOINT_EVERY == 0:
                save_checkpoint(dataset)

        save_checkpoint(dataset)

    # ── Also embed existing Gemini P5E results ──────────────────────
    if "gemini-2.0-flash" not in dataset["humanized"]:
        if p5e_file.exists():
            with open(p5e_file) as f:
                gemini_data = json.load(f)
            if "P5E_enhanced_lexicon" in gemini_data.get("humanized", {}):
                dataset["humanized"]["gemini-2.0-flash"] = gemini_data["humanized"]["P5E_enhanced_lexicon"]
                dataset["metadata"]["models"]["gemini-2.0-flash"] = "Gemini-2.0-Flash"
                print(f"\nEmbedded Gemini P5E results ({sum(1 for x in dataset['humanized']['gemini-2.0-flash'] if x)} texts)")

    # ── Save final ──────────────────────────────────────────────────
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    for key in dataset["humanized"]:
        done = sum(1 for x in dataset["humanized"][key] if x is not None)
        failed = N_TEXTS - done
        status = "COMPLETE" if failed == 0 else f"{failed} MISSING"
        print(f"  {key}: {done}/{N_TEXTS} ({status})")
    print(f"\nSaved to: {OUTPUT_FILE}")
    print("Upload this file to Kaggle for scoring.")


if __name__ == "__main__":
    main()
