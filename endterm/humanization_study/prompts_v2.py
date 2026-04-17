"""
Round 2 Humanization Prompts — Designed to beat Falcon-7B-Instruct
==================================================================
Based on analysis of Round 1 results:
- P5 (Anti-AI Lexicon) won: 78.3% gap closed, AUROC 0.731
- P2 (Burstiness) was #2: 69.5% gap closed
- P6 (Kitchen Sink) underperformed: too many instructions diluted the effect
- Key: Falcon-7B-Instruct predicts AI text well because AI text uses
  high-probability tokens and formulaic patterns. We need to increase
  perplexity by forcing unusual tokens and structures.

Strategy: 4 new prompts, each precisely targeting Falcon's prediction mechanism.
"""

PROMPTS_V2 = {

    # P7 — Focused P5+P2 hybrid (only the two winning strategies, tightly combined)
    # Why: P5 wins 60/100 texts, P2 wins 39/100. Together = 90% theoretical ceiling.
    # P6 failed because 10 instructions diluted focus. P7 has only 3 clear rules.
    "P7_lexicon_plus_structure": (
        "Rewrite this text following these THREE rules strictly:\n\n"
        "1. NEVER use these words: delve, crucial, pivotal, robust, leverage, "
        "facilitate, utilize, comprehensive, furthermore, moreover, additionally, "
        "realm, tapestry, landscape, testament, foster, underscore, multifaceted, "
        "intricate, nuanced, noteworthy, commendable, meticulous, paramount, "
        "indispensable, embark, embrace, significant, essential, innovative, "
        "demonstrate, emphasize, highlight, implement, enhance. "
        "Replace them with plain everyday words.\n\n"
        "2. VARY sentence length dramatically: alternate between very short "
        "sentences (3-8 words) and long complex ones (30+ words). Never put "
        "two sentences of similar length next to each other.\n\n"
        "3. NEVER use parallel structures or lists of three. "
        "Break any pattern where consecutive sentences follow the same format.\n\n"
        "Same meaning, same approximate length. "
        "Output only the rewritten text.\n\n"
        "Text to rewrite:\n{text}"
    ),

    # P8 — Extreme lexical unpredictability
    # Why: Falcon predicts AI text because AI uses high-probability tokens.
    # Force the LEAST expected word at every opportunity → directly increases perplexity.
    "P8_extreme_lexical": (
        "Rewrite this text using the most unexpected vocabulary possible while "
        "keeping the meaning intact. At every opportunity, choose a less common "
        "synonym over the obvious word. For example:\n"
        "- 'important' → 'load-bearing' or 'non-negotiable'\n"
        "- 'said' → 'let slip' or 'fired back'\n"
        "- 'increase' → 'spike' or 'balloon'\n"
        "- 'problem' → 'headache' or 'mess'\n"
        "- 'help' → 'bail out' or 'prop up'\n"
        "- 'significant' → 'hefty' or 'eye-popping'\n\n"
        "Also: avoid any word that appears more than once. Use contractions. "
        "Never start two sentences the same way. "
        "The text should feel like a sharp, idiosyncratic journalist wrote it — "
        "someone with a distinctive vocabulary, not a generic voice.\n\n"
        "Same facts, same approximate length. "
        "Output only the rewritten text.\n\n"
        "Text to rewrite:\n{text}"
    ),

    # P9 — Anti-instruction pattern (make it NOT look like AI output)
    # Why: Falcon-7B-Instruct was fine-tuned on instruction-following data.
    # It predicts AI text well because AI text LOOKS like instruction output.
    # If we remove all instruction-output markers, Falcon can't predict it.
    "P9_anti_instruction": (
        "Rewrite this text so it reads like someone's raw first draft — "
        "written quickly, not in response to any prompt or assignment. "
        "Apply these rules:\n\n"
        "- Remove ALL organizational markers: no 'firstly', 'in conclusion', "
        "'it is worth noting', 'as mentioned', 'in summary'\n"
        "- Remove ALL hedging phrases: no 'it is important to', "
        "'one might argue', 'it should be noted'\n"
        "- Use sentence fragments where natural. Not every sentence needs "
        "a subject and verb.\n"
        "- Start at least 2 sentences with 'And' or 'But'\n"
        "- Include one mid-sentence correction: 'well, actually' or "
        "'— no, wait —' or 'or rather'\n"
        "- Use at least 2 em dashes (—) and 1 semicolon\n"
        "- Write one deliberately short paragraph (1 sentence only)\n"
        "- NO topic sentences. Jump straight into details.\n\n"
        "Same information, same approximate length. "
        "Output only the rewritten text.\n\n"
        "Text to rewrite:\n{text}"
    ),

    # P10 — Structural noise + rare punctuation
    # Why: AI text has clean, predictable punctuation and structure.
    # Human text has interruptions, asides, varied punctuation.
    # These are LOW probability tokens for Falcon → high perplexity.
    "P10_structural_noise": (
        "Rewrite this text injecting natural human messiness into the structure. "
        "You MUST include ALL of these elements:\n\n"
        "- At least 3 parenthetical asides (in parentheses)\n"
        "- At least 2 em dashes (—) for interrupted thoughts\n"
        "- At least 1 semicolon connecting related ideas\n"
        "- At least 1 rhetorical question\n"
        "- At least 1 one-word or two-word sentence for emphasis\n"
        "- Mix paragraph lengths: one very short (1-2 sentences), "
        "one longer (4-5 sentences)\n"
        "- Use contractions throughout (don't, it's, won't, they're)\n"
        "- Start one sentence with 'Look,' and another with 'Thing is,'\n"
        "- Avoid these AI words: crucial, comprehensive, robust, utilize, "
        "facilitate, leverage, furthermore, moreover, delve, tapestry, "
        "landscape, testament, foster, underscore, pivotal\n\n"
        "Same facts, same approximate length. "
        "Output only the rewritten text.\n\n"
        "Text to rewrite:\n{text}"
    ),
}

PROMPT_LABELS_V2 = {
    "P7_lexicon_plus_structure": "P7: Lexicon + Structure (P5+P2 Hybrid)",
    "P8_extreme_lexical": "P8: Extreme Lexical Unpredictability",
    "P9_anti_instruction": "P9: Anti-Instruction Pattern",
    "P10_structural_noise": "P10: Structural Noise Injection",
}
