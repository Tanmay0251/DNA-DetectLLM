import numpy as np
import torch
import transformers
import random
from torch.nn import functional as F

ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
softmax_fn = torch.nn.Softmax(dim=-1)


@torch.no_grad()
def sum_perplexity(encoding: transformers.BatchEncoding,
                   logits: torch.Tensor,
                   median: bool = False,
                   temperature: float = 1.0):
    shifted_logits = logits[..., :-1, :] / temperature
    attention = encoding.attention_mask[..., 1:]
    labels_std = encoding.input_ids[..., 1:]
    labels_max = torch.argmax(shifted_logits, dim=-1)

    logits_T = shifted_logits.transpose(1, 2)

    ce_std = F.cross_entropy(logits_T, labels_std, reduction='none')
    ce_max = F.cross_entropy(logits_T, labels_max, reduction='none')

    attn_sum = attention.sum(dim=1).clamp(min=1)
    ppl_std = (ce_std * attention).sum(dim=1) / attn_sum
    ppl_max = (ce_max * attention).sum(dim=1) / attn_sum

    return (ppl_std + ppl_max).cpu().float().numpy()


def entropy(p_logits: torch.Tensor,
            q_logits: torch.Tensor,
            encoding: transformers.BatchEncoding,
            pad_token_id: int,
            median: bool = False,
            sample_p: bool = False,
            temperature: float = 1.0):
    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]
    p_scores, q_scores = p_logits / temperature, q_logits / temperature

    p_proba = softmax_fn(p_scores).view(-1, vocab_size)

    if sample_p:
        p_proba = torch.multinomial(p_proba.view(-1, vocab_size), replacement=True, num_samples=1).view(-1)

    q_scores = q_scores.view(-1, vocab_size)

    ce = ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available)
    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

    if median:
        ce_nan = ce.masked_fill(~padding_mask.bool(), float("nan"))
        agg_ce = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        agg_ce = (((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy())

    return agg_ce


def perplexity(encoding: transformers.BatchEncoding,
               logits: torch.Tensor,
               median: bool = False,
               temperature: float = 1.0):
    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()
    if median:
        ce_nan = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels).
                  masked_fill(~shifted_attention_mask.bool(), float("nan")))
        ppl = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        ppl = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) *
               shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
        ppl = ppl.to("cpu").float().numpy()
    return ppl


def auc_perplexity(encoding: transformers.BatchEncoding,
                   logits: torch.Tensor,
                   median: bool = False,
                   temperature: float = 1.0,
                   max_batch_size: int = 50,
                   repair_order: str = "s"):
    shifted_logits = logits[..., :-1, :] / temperature
    shifted_attention_mask = encoding.attention_mask[..., 1:]

    probs = torch.softmax(shifted_logits, dim=-1)
    max_prob_tokens = probs.argmax(dim=-1)

    input_ids = encoding.input_ids[..., 1:].clone()
    current_labels = input_ids.clone()

    ce_initial = ce_loss_fn(shifted_logits.transpose(1, 2), current_labels).float()
    ppl_sequence = [(ce_initial * shifted_attention_mask).sum() / shifted_attention_mask.sum()]

    logits_diff = torch.abs(
        probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
        - probs.gather(-1, max_prob_tokens.unsqueeze(-1)).squeeze(-1)
    )

    non_max_mask = (input_ids != max_prob_tokens)
    change_indices = non_max_mask.nonzero(as_tuple=False)

    if repair_order == "s":
        sorted_indices = change_indices
    elif repair_order == "h2l":
        sorted_indices = sorted(change_indices.tolist(), key=lambda idx: logits_diff[idx[0], idx[1]].item())
    elif repair_order == "l2h":
        sorted_indices = sorted(change_indices.tolist(), key=lambda idx: -logits_diff[idx[0], idx[1]].item())
    elif repair_order == "r":
        sorted_indices = change_indices.tolist()
        random.shuffle(sorted_indices)

    for idx in change_indices:
        batch_idx, token_idx = idx
        current_labels[batch_idx, token_idx] = max_prob_tokens[batch_idx, token_idx]
        ce_current = ce_loss_fn(shifted_logits.transpose(1, 2), current_labels)
        current_ppl = (ce_current * shifted_attention_mask).sum() / shifted_attention_mask.sum()
        current_ppl = current_ppl.float()
        ppl_sequence.append(current_ppl)

    ppl_sequence_np = torch.stack(ppl_sequence).cpu().numpy()
    auc = np.mean(ppl_sequence_np)
    return auc
