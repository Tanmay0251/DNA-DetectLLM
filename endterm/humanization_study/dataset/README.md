# Humanization Datasets

All JSON files referenced by the end-term scoring notebooks. Source texts come from public datasets (XSum, M4); humanized outputs were generated via the Groq API free tier using the prompts in `../prompts*.py`.

Total size ~14 MB.

## Core datasets (used by results/ notebooks)

| File | Size | Source | Humanizer | Prompt | Used by |
|------|------|--------|-----------|--------|---------|
| `gpt4_base_texts.json` | 208 KB | XSum | none (original GPT-4) | — | baseline |
| `base_texts_200.json` | 876 KB | XSum | none | — | prompt exploration |
| `humanization_dataset.json` | 1.6 MB | XSum | Gemini-2.0-Flash | P1-P6 | `score_humanization_v1` |
| `humanization_dataset_v2.json` | 1.9 MB | XSum | Gemini-2.0-Flash | P1-P6 + PA, PB | `score_humanization_v2` |
| `kimi_p1p6_dataset.json` | 644 KB | XSum | Kimi-K2 | P1-P6 + PA, PB | `score-p1p6-rfinal` |
| `p5e_dataset.json` | 588 KB | XSum | Gemini-2.0-Flash | P5E | P5E experiments |
| `multimodel_p5e_dataset.json` | 1.5 MB | XSum | 6 humanizers | P5E | `score-multi-model-p5e-humanization`, `score-mistral-detector`, `score-llama2-det` |
| `m4_literature_dataset.json` | 588 KB | M4 benchmark | Kimi-K2 | P_LIT3 | `m4-triplet-falcon`, `m4-triplet-mistral` |
| `m4_lit3_kimi.json` | 124 KB | M4 benchmark | Kimi-K2 | P_LIT3 | `score-m4-lit3-falcon` |
| `benchmark_dataset.json` | 2.5 MB | M4 + DetectRL + RealDet + XSum | — | — | `score-rfinal-benchmark` |

## Per-model runs (individual humanizer outputs)

| File | Size | Humanizer |
|------|------|-----------|
| `model_moonshotai_kimi-k2-instruct.json` | 164 KB | Kimi-K2 |
| `model_qwen_qwen3-32b.json` | 192 KB | Qwen-3-32B |
| `model_llama-3.1-8b-instant.json` | 204 KB | Llama-3.1-8B |
| `model_meta-llama_llama-4-scout-17b-16e-instruct.json` | 184 KB | Llama-4-Scout-17B |

## Intermediate checkpoints (kept for reproducibility)

| File | Size | What it is |
|------|------|-----------|
| `pilot_results.json` | 264 KB | Pilot run before scaling to 100 samples |
| `r3_results.json` | 156 KB | P5E (P5-revision-3) run snapshot |
| `raw_results.json` | 1.2 MB | Raw multi-model humanization output before filtering |
| `final_raw_results.json` | 352 KB | Filtered final output |
| `multimodel_checkpoint.json` | 652 KB | Resumable checkpoint from `generate_multimodel.py` |

## Schema

Each humanized record looks like:
```json
{
  "id": "xsum_42",
  "original_text": "...GPT-4 generated text...",
  "humanized_text": "...rewritten by the humanizer...",
  "humanizer_model": "moonshotai/kimi-k2-instruct",
  "prompt_name": "P_LIT3"
}
```

Baseline files (`gpt4_base_texts.json`, `base_texts_200.json`) contain only original texts:
```json
{"id": "...", "text": "...", "label": "human" | "ai"}
```

## Licensing

- Original XSum texts: Creative Commons (BBC News)
- M4 benchmark texts: per the M4 paper's release
- Humanized outputs: generated via third-party LLM APIs; redistributed under academic fair use for this coursework
