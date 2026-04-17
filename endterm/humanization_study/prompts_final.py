"""
FINAL Humanization Prompts — designed after deep analysis of Round 1 results.
=============================================================================

Key insight: We need tokens that are NATURAL but UNEXPECTED for Falcon-7B-Instruct.
- Not "plain/simple" (also predictable)
- Not "rare/unusual" (sounds forced)
- Sweet spot: "vivid/specific/concrete" — natural but not the first guess

Two approaches:
  PA: Rule-based (ban AI words + force vivid vocabulary + structure variation)
  PB: Persona-based (write as a specific journalist → naturally produces all desired traits)
"""

PROMPTS_FINAL = {

    # PA — Vivid Lexicon + Structure (refined from P5+P2 analysis)
    "PA_vivid_lexicon": (
        "Rewrite this text following these rules:\n\n"
        "VOCABULARY: Never use these words — delve, crucial, pivotal, robust, "
        "leverage, facilitate, utilize, comprehensive, furthermore, moreover, "
        "additionally, realm, tapestry, landscape, testament, foster, underscore, "
        "multifaceted, intricate, nuanced, noteworthy, meticulous, paramount, "
        "indispensable, embark, embrace, significant, essential, innovative, "
        "demonstrate, emphasize, highlight, implement, enhance, ensure, "
        "transformative, endeavor, bolster. "
        "Replace every generic word with a VIVID, CONCRETE alternative. "
        "Don't just simplify — use specific, colorful language. "
        "Example: 'significant increase' → 'a sharp spike'; "
        "'implemented a solution' → 'patched the problem'; "
        "'comprehensive analysis' → 'a deep dive'.\n\n"
        "STRUCTURE: Alternate between punchy short sentences (3-8 words) and "
        "longer complex ones (25+ words). Never put two same-length sentences "
        "next to each other. Break all parallel patterns.\n\n"
        "Same facts, same approximate length. "
        "Output only the rewritten text.\n\n"
        "Text to rewrite:\n{text}"
    ),

    # PB — Sharp Journalist Persona
    "PB_journalist_persona": (
        "You are a veteran newspaper columnist known for your sharp, distinctive "
        "writing voice. You never sound like a press release or a textbook. "
        "Your style: punchy, opinionated, vivid. You favor concrete details over "
        "abstractions, short declarative sentences mixed with long flowing ones, "
        "and you're allergic to corporate or academic jargon.\n\n"
        "Rewrite the text below in YOUR voice. Make it sound like you dashed it off "
        "on a tight deadline — confident, slightly informal, full of personality. "
        "Use contractions. Throw in a rhetorical question if it fits. "
        "Never use words like 'crucial', 'comprehensive', 'utilize', 'facilitate', "
        "'robust', 'leverage', 'furthermore', 'moreover', 'delve', 'tapestry', "
        "'landscape', 'foster', 'underscore', 'pivotal', or 'testament'.\n\n"
        "Same facts, same approximate length. "
        "Output only the rewritten text.\n\n"
        "Text to rewrite:\n{text}"
    ),
}

PROMPT_LABELS_FINAL = {
    "PA_vivid_lexicon": "PA: Vivid Lexicon + Structure",
    "PB_journalist_persona": "PB: Sharp Journalist Persona",
}
