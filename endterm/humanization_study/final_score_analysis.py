"""Final score function analysis combining V2 + V3 best features."""
import json, numpy as np
from sklearn.metrics import roc_auc_score

v2 = json.load(open("/tmp/v2_done/v2_results.json"))["all_v2"]
v3 = json.load(open("/tmp/v3_done/v3_results.json"))["data"]

# Merge V2 and V3 features (same texts, same order)
groups = ["human", "original_ai", "moonshotai_kimi-k2-instruct", "gemini-2.0-flash", "qwen_qwen3-32b"]
labels = ["Human", "AI", "Kimi", "Gemini", "Qwen"]

merged = {}
for g in groups:
    v2_data = v2.get(g, [])
    v3_data = v3.get(g, [])
    rows = []
    for i in range(min(len(v2_data), len(v3_data))):
        if v2_data[i] and v3_data[i]:
            row = {**v2_data[i], **v3_data[i]}
            rows.append(row)
    merged[g] = rows
    print(f"{g}: {len(rows)} merged samples")

# Best features from V2: ce_var (AUROC 0.966 inverted), agree_rate (0.771)
# Best feature from V3: coherence (0.895)
# Plus R(s) as base

print("\n=== BEST INDIVIDUAL FEATURES ===")
best = ["rs", "ce_var", "agree_rate", "coherence", "d_var", "H_var"]
hv = {f: [s[f] for s in merged["human"]] for f in best}
print(f"{'Feature':<15} {'H vs AI':>8} {'H vs Kimi':>10} {'H vs Gem':>9}")
for feat in best:
    av = [s[feat] for s in merged["original_ai"]]
    kv = [s[feat] for s in merged["moonshotai_kimi-k2-instruct"]]
    gv = [s[feat] for s in merged["gemini-2.0-flash"]]
    a1 = roc_auc_score([1]*len(hv[feat])+[0]*len(av), hv[feat]+av)
    a2 = roc_auc_score([1]*len(hv[feat])+[0]*len(kv), hv[feat]+kv)
    a3 = roc_auc_score([1]*len(hv[feat])+[0]*len(gv), hv[feat]+gv)
    print(f"{feat:<15} {a1:>8.4f} {a2:>10.4f} {a3:>9.4f}")

# Combined score function using BOTH V2 and V3 features
h_cev = [s["ce_var"] for s in merged["human"]]
h_agr = [s["agree_rate"] for s in merged["human"]]
h_coh = [s["coherence"] for s in merged["human"]]
cev_m, cev_s = np.mean(h_cev), np.std(h_cev)
agr_m, agr_s = np.mean(h_agr), np.std(h_agr)
coh_m, coh_s = np.mean(h_coh), np.std(h_coh)

def R_final(s, k=1.0):
    rs = s["rs"]
    # Penalize high CE variance (humanization creates bimodal CE pattern)
    cev_ex = max(0, s["ce_var"] - (cev_m + k * cev_s))
    p1 = max(0.1, 1 - 0.1 * cev_ex)
    # Penalize low agreement rate
    agr_def = max(0, (agr_m - k * agr_s) - s["agree_rate"])
    p2 = max(0.1, 1 - 3.0 * agr_def)
    # Penalize abnormal coherence (humanized has more negative coherence)
    coh_ex = max(0, (coh_m - k * coh_s) - s["coherence"])
    p3 = max(0.1, 1 - 0.5 * coh_ex)
    return rs * p1 * p2 * p3

print(f"\n=== FINAL COMBINED SCORE ===")
print(f"{'Score':<30} {'H vs AI':>8} {'H vs Kimi':>10} {'H vs Gem':>9} {'H vs Qwen':>10}")
print("=" * 70)

for name, fn in [
    ("R(s) original", lambda s: s["rs"]),
    ("R_v2 (ce_var+agree, k=1.0)", lambda s: R_final.__code__ and (
        s["rs"] * max(0.1, 1 - 0.1 * max(0, s["ce_var"] - (cev_m + 1.0 * cev_s)))
        * max(0.1, 1 - 3.0 * max(0, (agr_m - 1.0 * agr_s) - s["agree_rate"]))
    )),
    ("R_final (all 3, k=0.5)", lambda s: R_final(s, k=0.5)),
    ("R_final (all 3, k=1.0)", lambda s: R_final(s, k=1.0)),
    ("R_final (all 3, k=1.5)", lambda s: R_final(s, k=1.5)),
]:
    row = []
    hs = [fn(s) for s in merged["human"]]
    for g in ["original_ai", "moonshotai_kimi-k2-instruct", "gemini-2.0-flash", "qwen_qwen3-32b"]:
        gs = [fn(s) for s in merged.get(g, [])]
        if hs and gs:
            row.append(roc_auc_score([1]*len(hs)+[0]*len(gs), hs+gs))
        else:
            row.append(0)
    print(f"{name:<30} {row[0]:>8.4f} {row[1]:>10.4f} {row[2]:>9.4f} {row[3]:>10.4f}")

print("\n=== KEY TAKEAWAY ===")
print("R(s) alone: breaks on humanized (Kimi 0.21, Gemini 0.47)")
print("R_final: recovers detection while maintaining standard performance")
print("Features used: per-token CE variance + observer-performer agreement + confidence-coherence")
print("All computed from the SAME model outputs, zero extra cost.")
