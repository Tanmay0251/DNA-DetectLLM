"""
6 Humanization Prompts for DNA-DetectLLM Robustness Study
=========================================================
Each prompt targets a different weakness of perplexity-based AI detectors.
Applied to GPT-4 machine text, self-humanized via Gemini-2.0-Flash.
"""

PROMPTS = {

    # P1 — Baseline: minimal rewrite, just paraphrase
    "P1_simple_paraphrase": (
        "Rewrite the following text to sound more natural and human-written. "
        "Keep the same meaning and approximate length. "
        "Output only the rewritten text, nothing else.\n\n"
        "Text to rewrite:\n{text}"
    ),

    # P2 — Burstiness: attack statistical sentence-length uniformity
    "P2_burstiness": (
        "Rewrite this text with dramatically varied sentence lengths. "
        "Mix very short punchy sentences (3-7 words) with longer complex ones (25-40 words). "
        "Never have two consecutive sentences of similar length. "
        "Restructure paragraphs so they are uneven in size. "
        "Same meaning, same approximate total length. "
        "Output only the rewritten text, nothing else.\n\n"
        "Text to rewrite:\n{text}"
    ),

    # P3 — Personal voice: inject opinions, hedging, first-person
    "P3_personal_voice": (
        "Rewrite this text as if you are a journalist with strong personal opinions. "
        "Add hedging language like 'I think', 'arguably', 'it seems to me', 'in my view'. "
        "Include personal observations and occasional first-person commentary. "
        "Add at least one rhetorical question. "
        "Same core facts, same approximate length. "
        "Output only the rewritten text, nothing else.\n\n"
        "Text to rewrite:\n{text}"
    ),

    # P4 — Conversational: shift register to informal spoken style
    "P4_conversational": (
        "Rewrite this text in a casual, conversational tone — like you're explaining it "
        "to a friend over coffee. Use contractions freely (don't, it's, won't, can't). "
        "Start some sentences with 'And' or 'But'. Use informal transitions like "
        "'here's the thing', 'look', 'honestly', 'so basically'. "
        "Throw in a rhetorical question or two. "
        "Same information, same approximate length. "
        "Output only the rewritten text, nothing else.\n\n"
        "Text to rewrite:\n{text}"
    ),

    # P5 — Anti-AI lexicon: remove known AI-flagged words and patterns
    "P5_anti_ai_lexicon": (
        "Rewrite this text while strictly avoiding these AI-associated words and patterns:\n"
        "BANNED WORDS: delve, crucial, pivotal, robust, leverage, facilitate, utilize, "
        "comprehensive, furthermore, moreover, additionally, realm, tapestry, landscape, "
        "testament, foster, underscore, multifaceted, intricate, nuanced, noteworthy, "
        "commendable, meticulous, paramount, indispensable, embark, embrace.\n"
        "BANNED PATTERNS: Do not use parallel sentence structures. Do not start paragraphs "
        "with a topic sentence followed by supporting details. Avoid tricolon lists (lists of three). "
        "Never use 'It is important to note' or 'In today's world' or 'In conclusion'.\n"
        "Replace any banned element with plain, everyday language. "
        "Same meaning, same approximate length. "
        "Output only the rewritten text, nothing else.\n\n"
        "Text to rewrite:\n{text}"
    ),

    # P6 — Kitchen sink: maximum aggression, all techniques combined
    "P6_kitchen_sink": (
        "You are an expert writer. Your task is to completely rewrite the following text "
        "so it is indistinguishable from human writing.\n\n"
        "Apply ALL of these techniques simultaneously:\n"
        "- Vary sentence length wildly (mix 5-word and 40-word sentences)\n"
        "- Use contractions and colloquialisms freely (don't, it's, honestly, look)\n"
        "- Add personal opinions: 'I think', 'arguably', 'in my view'\n"
        "- Include at least one rhetorical question\n"
        "- Break formulaic paragraph structures completely\n"
        "- Use concrete sensory details instead of abstract summaries\n"
        "- Never start two consecutive sentences the same way\n"
        "- Bend a grammar rule or two naturally (fragment, dangling modifier)\n"
        "- Add a brief tangential aside that a human would naturally include\n"
        "- Avoid these AI words: delve, crucial, robust, comprehensive, utilize, furthermore, "
        "moreover, tapestry, landscape, facilitate, leverage, pivotal\n\n"
        "The result must read as if a specific journalist dashed it off on deadline. "
        "Same core facts, same approximate length. "
        "Output only the rewritten text, nothing else.\n\n"
        "Text to rewrite:\n{text}"
    ),
}

# Short descriptions for labeling
PROMPT_LABELS = {
    "P1_simple_paraphrase": "Simple Paraphrase",
    "P2_burstiness": "Burstiness & Structure",
    "P3_personal_voice": "Personal Voice & Opinions",
    "P4_conversational": "Conversational Tone",
    "P5_anti_ai_lexicon": "Anti-AI Lexicon",
    "P6_kitchen_sink": "Kitchen Sink (All Combined)",
}
