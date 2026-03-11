# DNA-DetectLLM Replication

IE 663 Course Project - Team Eternal (Tanmay Mandaliya, 22B1037)

Replication and analysis of [DNA-DetectLLM](https://arxiv.org/abs/2509.15550) (NeurIPS 2025) on Kaggle T4x2 GPUs with 4-bit NF4 quantization.

## Notebooks

- `replication-and-analysis.ipynb` - Main replication notebook covering Tables 1, 2, 10 along with component ablation, temperature sensitivity, and short text analysis
- `demo-notebook.ipynb` - Quick demo showing how DNA-DetectLLM works on a few sample texts

## Code

- `dna_detectllm/` - The paper's official detection code (detector, metrics, utils) with a minor modification for 4-bit NF4 quantization so it fits on T4 GPUs

## How to run

1. Upload the notebook you want to run to Kaggle
2. Zip the `dna_detectllm/` directory and upload it as a Kaggle dataset
3. Update the path in the notebook cell that copies `dna_detectllm` to point to your uploaded dataset path (e.g. `/kaggle/input/your-dataset-name/dna_detectllm`)
4. Make sure you have 2x T4 GPUs enabled in the Kaggle notebook settings
5. Run all cells to replicate the experiments
