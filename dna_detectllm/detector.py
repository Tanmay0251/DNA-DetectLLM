from typing import Union

import os
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from .utils import assert_tokenizer_consistency
from .metrics import sum_perplexity, entropy, auc_perplexity

torch.set_grad_enabled(False)

# Thresholds selected using Falcon-7B and Falcon-7B-Instruct
# Original paper (bfloat16): 0.9015 (accuracy), 0.8536 (low-FPR)
# Recalibrated for 4-bit NF4 quantization on T4 GPUs:
detectllm_ACCURACY_THRESHOLD = 0.64
detectllm_FPR_THRESHOLD = 0.58

DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1


class DetectLLM(object):
    def __init__(self,
                 observer_name_or_path: str = "tiiuae/falcon-7b",
                 performer_name_or_path: str = "tiiuae/falcon-7b-instruct",
                 use_4bit: bool = True,
                 max_token_observed: int = 1024,
                 mode: str = "low-fpr",
                 ) -> None:
        assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)

        self.change_mode(mode)

        # 4-bit NF4 quantization to fit on T4 GPUs
        if use_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            quant_config = None

        load_kwargs = dict(low_cpu_mem_usage=True)
        if quant_config:
            load_kwargs["quantization_config"] = quant_config
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        self.observer_model = AutoModelForCausalLM.from_pretrained(
            observer_name_or_path,
            device_map={"": DEVICE_1},
            **load_kwargs,
        )
        self.performer_model = AutoModelForCausalLM.from_pretrained(
            performer_name_or_path,
            device_map={"": DEVICE_2},
            **load_kwargs,
        )
        self.observer_model.eval()
        self.performer_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed

    def change_mode(self, mode: str) -> None:
        if mode == "low-fpr":
            self.threshold = detectllm_FPR_THRESHOLD
        elif mode == "accuracy":
            self.threshold = detectllm_ACCURACY_THRESHOLD
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False).to(self.observer_model.device)
        return encodings

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding) -> torch.Tensor:
        observer_logits = self.observer_model(**encodings.to(DEVICE_1)).logits
        performer_logits = self.performer_model(**encodings.to(DEVICE_2)).logits
        if DEVICE_1 != "cpu":
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    def cleanup(self):
        if self.observer_model is not None:
            del self.observer_model
            del self.performer_model
            self.observer_model = None
            self.performer_model = None
        torch.cuda.empty_cache()

    def compute_score(self, input_text: Union[list[str], str]) -> Union[float, list[float]]:
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)

        ppl = sum_perplexity(encodings, performer_logits)
        x_ppl = entropy(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
                        encodings.to(DEVICE_1), self.tokenizer.pad_token_id)

        detectllm_scores = ppl / (2 * x_ppl)
        detectllm_scores = detectllm_scores.tolist()

        return detectllm_scores

    def compute_score_iterative(self, input_text: Union[list[str], str],
                                repair_order: str = "h2l") -> Union[float, list[float]]:
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)

        ppl_auc = auc_perplexity(encodings, performer_logits, repair_order=repair_order)
        x_ppl = entropy(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
                        encodings.to(DEVICE_1), self.tokenizer.pad_token_id)

        detectllm_scores = ppl_auc / x_ppl
        detectllm_scores = detectllm_scores.tolist()

        return detectllm_scores

    def predict(self, input_text: Union[list[str], str]) -> Union[list[str], str]:
        detectllm_scores = np.array(self.compute_score(input_text))
        pred = np.where(detectllm_scores < self.threshold,
                        "Most likely AI-generated",
                        "Most likely human-generated"
                        ).tolist()
        return pred
