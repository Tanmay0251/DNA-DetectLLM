# DNA-DetectLLM Replication

IE 663 Course Project - Team Eternal (Tanmay Mandaliya, 22B1037)

Replication and analysis of [DNA-DetectLLM](https://arxiv.org/abs/2509.15550) (NeurIPS 2025) on Kaggle T4x2 GPUs with 4-bit NF4 quantization.

## Notebooks

- `replication-and-analysis.ipynb` - Main replication notebook covering Tables 1, 2, 10 along with component ablation, temperature sensitivity, and short text analysis
- `demo-notebook.ipynb` - Quick demo showing how DNA-DetectLLM works on a few sample texts

Both notebooks were run on Kaggle with 2x T4 GPUs. They pull data from the [official repo](https://github.com/Xiaoweizhu57/DNA-DetectLLM) and use Falcon-7B + Falcon-7B-Instruct loaded in 4-bit NF4 quantization.
