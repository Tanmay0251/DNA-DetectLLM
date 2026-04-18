# DNA-DetectLLM: Replication, Humanization Study, and R_final

IE 663 Course Project — IIT Bombay — Team Eternal (Tanmay Mandaliya, 22B1037)

This repository contains the midterm replication and the end-term extension of [DNA-DetectLLM](https://arxiv.org/abs/2509.15550) (NeurIPS 2025). The end-term work shows that the detector can be broken by a carefully designed humanization prompt, and proposes an extension (`R_final`) that recovers detection on unseen data.

---

## Repository Layout

```
.
├── replication-and-analysis.ipynb     # Midterm: Tables 1, 2, 10 + ablations
├── dna_detectllm/                     # Paper's official code (4-bit patched)
│
└── endterm/
    ├── demo_notebook.ipynb            # Live demo: R(s) fooled, R_final catches
    ├── humanization_study/            # All end-term Python code
    │   ├── prompts.py                 # P1-P6 + PA, PB humanization prompts
    │   ├── prompts_r3.py              # P5E (data-driven enhanced prompt)
    │   ├── prompts_literature.py      # P_LIT1/2/3 (literature-based prompts)
    │   ├── word_frequency_analysis.py # How P5E's banned word list was derived
    │   ├── p5e_cross_domain_failure.py# Shows P5E is overfit to XSum (49.5% leak on ArXiv)
    │   ├── gen_single_model.py        # Groq API wrapper for humanization
    │   ├── generate_multimodel.py     # Multi-humanizer pipeline
    │   ├── final_score_analysis.py    # R_final penalty tuning and ablation
    │   └── ...
    │
    ├── results/                       # 10 experiment notebooks (with full outputs)
    │   ├── score-multi-model-p5e-humanization.ipynb   # 6 humanizers on Falcon
    │   ├── score-mistral-detector.ipynb               # Cross-detector: Mistral
    │   ├── score-llama2-det.ipynb                     # Cross-detector: Llama-2
    │   ├── score-m4-lit3-falcon.ipynb                 # P_LIT3 on unseen M4 (Falcon)
    │   ├── m4-triplet-falcon.ipynb                    # R_final on M4 (Falcon)
    │   ├── m4-triplet-mistral.ipynb                   # R_final on M4 (Mistral)
    │   ├── score-v2-variance-features.ipynb          # 6 per-token features tested
    │   ├── score-v3-diveye.ipynb                      # 5 DivEye features tested
    │   ├── score-p1p6-rfinal.ipynb                    # R_final on 8 prompt strategies
    │   └── score-rfinal-benchmark.ipynb               # R_final on M4/DetectRL/RealDet/XSum
    │
    └── figures/                       # Plots used in report and slides
```

---

## Key Results

| Claim | Metric | Notebook |
|-------|--------|----------|
| Midterm replication | Avg AUROC 96.7% (paper 98.3%) | `replication-and-analysis.ipynb` |
| P5 wins among P1-P6 | AUROC 0.731 (XSum, Gemini) | `humanize_exp_results/` in midterm |
| P5E breaks Falcon on XSum | AUROC 0.538 | `score-multi-model-p5e-humanization.ipynb` |
| P5E leaks on ArXiv | 49.5% human texts contain banned words | `p5e_cross_domain_failure.py` |
| P_LIT3 breaks Falcon on unseen M4 | AUROC 0.144 | `score-m4-lit3-falcon.ipynb` |
| Cross-detector: Kimi fools all 3 | Falcon 0.197, Mistral 0.308, Llama-2 0.220 | `score-mistral-detector.ipynb`, `score-llama2-det.ipynb` |
| R_final recovers on Falcon M4 | 0.155 → 0.908 | `m4-triplet-falcon.ipynb` |
| R_final recovers on Mistral M4 | 0.475 → 0.945 | `m4-triplet-mistral.ipynb` |
| Standard benchmark cost | 5-9% AUROC drop | `score-rfinal-benchmark.ipynb` |

---

## R_final Score Function

```
R_final(s) = R(s) * P1(ce_var) * P2(agree_rate) * P3(coherence)
```

Three penalties are multiplied onto the original `R(s)`. Each stays at 1.0 when the feature is within the human baseline range and drops toward 0.1 when it deviates beyond k standard deviations.

| Feature | What it measures | Why it helps |
|---------|------------------|--------------|
| `ce_var` | Variance of per-token cross-entropy | Humanization creates a bimodal CE pattern (function words low, replaced content words very high) |
| `agree_rate` | Fraction of positions where observer and performer pick the same top-1 token | Humanized text has more observer-performer disagreement |
| `coherence` | Mean CE at low-entropy positions minus mean CE at high-entropy positions | Humanized text is surprising where the model is normally confident |

All three features come from the same two forward passes used by `R(s)` — zero extra inference cost.

Implementation: `humanization_study/final_score_analysis.py` (standalone) and inside every notebook in `endterm/results/` as the `R_final()` function.

---

## How to Run

### Midterm replication
1. Upload `replication-and-analysis.ipynb` to Kaggle
2. Zip `dna_detectllm/` and upload as a Kaggle dataset
3. Update the path in the first cell to point to your uploaded dataset (e.g. `/kaggle/input/your-dataset-name/dna_detectllm`)
4. Enable 2x T4 GPUs in Kaggle settings
5. Run all cells

### End-term demo (live on Kaggle)
1. Upload `endterm/demo_notebook.ipynb` to Kaggle
2. Attach Kaggle dataset: `mandaliyatanmay/dna-detectllm-orig` (contains pre-tokenized benchmark texts)
3. Add a Kaggle secret named `GROQ_API_KEY` (free tier at https://console.groq.com/keys)
4. Select a T4 (or P100) GPU
5. Run cells top to bottom. Cell 2 loads both Falcon models (~3 min cold). Cells 5-10 run the full demo:
   - Cell 5: score a human XSum text
   - Cell 6: score a GPT-4 text
   - Cell 7: show the P_LIT3 prompt
   - Cell 8: humanize the AI text via Kimi-K2 (Groq API)
   - Cell 9: score the humanized text (R(s) fooled, R_final catches)
   - Cell 10: summary table + aggregate AUROCs

The notebook has a pre-computed fallback humanized text, so Cell 9 still works even if Groq is unreachable.

### Running the experiment notebooks
Each notebook in `endterm/results/` is self-contained and was run on Kaggle P100 or T4. To reproduce:

1. Upload the notebook to Kaggle
2. Attach `mandaliyatanmay/dna-detectllm-orig` and the relevant humanized dataset listed at the top of each notebook
3. Set a T4 or P100 GPU
4. Run all cells

All notebooks are committed with their full outputs so you can read the results without rerunning anything.

### Generating new humanized text (optional)
```bash
cd endterm/humanization_study/
export GROQ_API_KEY="your_key_here"
python gen_single_model.py --model kimi-k2 --prompt P_LIT3 --input_json path/to/ai_texts.json --output_json humanized.json
```

See `generate_multimodel.py` for the batch version that runs all 6 humanizer models.

---

## Models Used

- **Detectors** (observer + performer pairs, 4-bit NF4 quantized):
  - Falcon-7B + Falcon-7B-Instruct (paper's main setup)
  - Mistral-7B + Mistral-7B-Instruct-v0.3
  - Llama-2-7B + Llama-2-7B-chat
- **Humanizers** (via Groq API free tier):
  - Kimi-K2 (moonshotai/kimi-k2-instruct)
  - Gemini-2.0-Flash
  - Qwen-3-32B
  - Llama-3.3-70B, Llama-3.1-8B, Llama-4-Scout-17B

---

## Citation

If you use this code, please cite the original paper:

```
Zhu, X., Ren, Y., Fang, F., Tan, Q., Wang, S., Cao, Y.
DNA-DetectLLM: Unveiling AI-Generated Text via a DNA-Inspired Mutation-Repair Paradigm.
NeurIPS 2025. https://arxiv.org/abs/2509.15550
```

---

## Course Context

Submitted for **IE 663: Advanced Topics in Deep Learning**, Spring 2026, IIT Bombay (Prof. P. Balamurugan). This is an academic replication and extension project. The end-term report and presentation are on the course Moodle.

All API keys have been scrubbed from the repository. Do not commit your own keys.
