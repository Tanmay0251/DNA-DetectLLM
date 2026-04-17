"""
Literature-Based Domain-Agnostic Humanization Prompts
=====================================================
These prompts are built ENTIRELY from published research on AI writing artifacts.
NO data was used to design these. The knowledge comes from:

1. Gehrmann et al. (GLTR, ACL 2019): AI tokens cluster in top-10 probability ranks
2. Mitchell et al. (DetectGPT, ICML 2023): AI text has low perturbation sensitivity
3. Hans et al. (Binoculars, ICML 2024): AI text has low cross-perplexity ratio
4. Zhu et al. (DNA-DetectLLM, NeurIPS 2025): AI text = greedy sequence + few mutations
5. General NLP knowledge: instruction-tuned models share vocabulary biases

Key insight: Perplexity-based detectors (DNA-DetectLLM, Binoculars, Fast-DetectGPT)
all rely on the same signal — AI text follows the model's probability distribution
too closely. To evade them, we need to INCREASE perplexity by injecting tokens the
model wouldn't predict, while keeping the text natural.

Three approaches, none using dataset-specific knowledge:

P_LIT1: Target the generation mechanism (inject low-probability tokens)
P_LIT2: Target structural patterns (instruction-tuning artifacts)
P_LIT3: Combined — both vocabulary and structure
"""

PROMPTS_LITERATURE = {

    # P_LIT1: Attack the token probability distribution
    # Rationale: AI detectors measure how predictable each token is.
    # If we replace predictable words with unpredictable-but-natural alternatives,
    # the detector sees higher perplexity = more human-like.
    "P_LIT1_token_surprise": (
        "Rewrite this text to sound like a seasoned journalist wrote it quickly. "
        "Follow these rules:\n\n"
        "1. WORD CHOICE: At every opportunity, choose a less obvious word. "
        "Replace any formal or academic word with a vivid, specific, informal alternative. "
        "Example: 'significant' → 'hefty'; 'demonstrated' → 'proved'; "
        "'implemented' → 'rolled out'; 'comprehensive' → 'wall-to-wall'; "
        "'facilitate' → 'grease the wheels'; 'utilize' → 'use'.\n\n"
        "2. CONTRACTIONS: Always use contractions (don't, it's, won't, they're, "
        "can't, shouldn't, we've). Never write 'do not' when 'don't' works.\n\n"
        "3. SENTENCE VARIETY: Alternate wildly between short punchy sentences "
        "(3-7 words) and long winding ones (25-40 words). Never put two "
        "similar-length sentences next to each other.\n\n"
        "Same facts, same approximate length. "
        "Output only the rewritten text.\n\n"
        "Text to rewrite:\n{text}"
    ),

    # P_LIT2: Attack instruction-tuning structural patterns
    # Rationale: All instruct models are trained on similar data (RLHF, ChatML).
    # They share structural habits: topic sentences, tricolon, formal transitions,
    # hedging phrases, parallel structures. These are predictable to the detector.
    "P_LIT2_structure_attack": (
        "Rewrite this text breaking ALL formulaic writing patterns:\n\n"
        "REMOVE these structures:\n"
        "- Never start a paragraph with a topic sentence followed by supporting details\n"
        "- Never use lists of three (tricolon) — use two items or four, never three\n"
        "- Never use transitions like 'furthermore', 'moreover', 'additionally', "
        "'in conclusion', 'it is worth noting', 'as a result', 'consequently'\n"
        "- Never use hedging: 'it is important to note', 'one might argue', "
        "'it should be noted', 'arguably'\n"
        "- Never use parallel sentence structures (where consecutive sentences "
        "follow the same grammatical pattern)\n\n"
        "ADD these human patterns:\n"
        "- Start at least 2 sentences with 'And' or 'But'\n"
        "- Use at least 1 sentence fragment (no subject or verb)\n"
        "- Include 1 parenthetical aside (in parentheses)\n"
        "- Use at least 1 em dash (—) for an interrupted thought\n"
        "- Write one very short paragraph (1-2 sentences only)\n"
        "- Use contractions throughout\n\n"
        "Same facts, same approximate length. "
        "Output only the rewritten text.\n\n"
        "Text to rewrite:\n{text}"
    ),

    # P_LIT3: Combined — maximum evasion
    # Uses both vocabulary surprise AND structural breaking
    "P_LIT3_combined": (
        "Rewrite this text so it reads like a specific human wrote it on a tight "
        "deadline — not like an AI responding to a prompt.\n\n"
        "VOCABULARY RULES:\n"
        "- Replace every formal/academic word with a vivid informal alternative. "
        "Don't just simplify — use SPECIFIC, COLORFUL language. "
        "Example: 'significant increase' → 'a sharp spike'; "
        "'implemented a solution' → 'patched the mess'; "
        "'various factors' → 'a tangle of reasons'.\n"
        "- Use contractions everywhere (don't, it's, won't, can't, they're).\n"
        "- Never repeat the same adjective or adverb twice.\n\n"
        "STRUCTURE RULES:\n"
        "- Alternate between very short sentences (3-7 words) and long ones (25+ words).\n"
        "- Never use topic-sentence-then-support paragraph structure.\n"
        "- Never use transitions: furthermore, moreover, additionally, consequently, "
        "in conclusion, it is worth noting, as a result.\n"
        "- Never use parallel structures or lists of three.\n"
        "- Include at least 1 sentence fragment, 1 parenthetical aside, "
        "and 1 em dash (—).\n"
        "- Start at least 1 sentence with 'But' or 'And'.\n\n"
        "Same facts, same approximate length. "
        "Output only the rewritten text.\n\n"
        "Text to rewrite:\n{text}"
    ),
}

PROMPT_LABELS_LITERATURE = {
    "P_LIT1_token_surprise": "LIT1: Token Surprise (vocabulary)",
    "P_LIT2_structure_attack": "LIT2: Structure Attack",
    "P_LIT3_combined": "LIT3: Combined (vocab + structure)",
}
