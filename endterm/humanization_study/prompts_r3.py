"""
Round 3: Data-driven P5 Enhancement
====================================
P5 bans 27 words but many don't appear in GPT-4 text (delve=0, tapestry=0, realm=0).
Meanwhile, highly AI-overrepresented words are NOT banned.

P5E: Same structure as P5 but with a DATA-DRIVEN banned word list
     based on actual word frequency analysis of GPT-4 vs human texts.
     Every banned word actually appears 5+ times in AI text and 0-1 times in human.
"""

PROMPTS_R3 = {

    # P5E — Enhanced Anti-AI Lexicon (data-driven banned list)
    "P5E_enhanced_lexicon": (
        "Rewrite this text while strictly avoiding these words and patterns:\n\n"
        "BANNED WORDS: crucial, comprehensive, testament, unwavering, numerous, "
        "determination, importance, unique, potentially, various, maintain, maintaining, "
        "additionally, resilience, ensuring, dedication, strategic, amidst, resulted, "
        "highlight, atmosphere, multiple, sustainable, broader, valuable, emotional, "
        "beloved, perspective, emphasized, implement, offerings, stance, outcomes, "
        "furthermore, moreover, pivotal, landscape, foster, underscore, paramount.\n\n"
        "BANNED PATTERNS: Do not use parallel sentence structures. Do not start "
        "paragraphs with a topic sentence followed by supporting details. "
        "Avoid lists of three. Never use 'It is important to note' or "
        "'It is worth mentioning' or 'In conclusion' or 'As a result'.\n\n"
        "Replace any banned word with a SPECIFIC, CONCRETE alternative — "
        "not a generic synonym. Same meaning, same approximate length. "
        "Output only the rewritten text.\n\n"
        "Text to rewrite:\n{text}"
    ),
}

PROMPT_LABELS_R3 = {
    "P5E_enhanced_lexicon": "P5E: Enhanced Anti-AI Lexicon (Data-Driven)",
}
