"""Constants and configuration defaults for the Unified Tree Engine."""

P_STOP_FLOOR: float = 1e-4
DEFAULT_OOD_MASS_THRESHOLD: float = 0.002
DEFAULT_OOD_PREMISE_THRESHOLD: float = -6.5

SUBWORD_BRANCH_FACTOR: int = 2
SUBWORD_BRANCH_RATIO: float = 0.15

DIAG_MODE: str = "fail"  # "off" | "fail" | "all"
DIAG_TOP_FIRST: int = 6
DIAG_TOP_NEXT: int = 5

IRREGULAR_STEM_EXCEPTIONS = {
    "string", "spring", "bring", "ring", "sing", "wing", "king", "thing",
    "bleed", "breed", "speed", "feed", "need", "seed",
}

# Known bound BPE stems in Qwen2.5 that are not standalone English words
KNOWN_BOUND_PREFIXES = {
    "emerg", "indemn", "card", "cardiol", "orth", "orthop", "neurol", "neuro",
    "hosp", "hospit", "frust", "frustr", "crit", "damag", "repl", "replac",
    "invest", "investig", "malw", "ransom", "ransomw", "phish", "infring",
    "sever", "severab", "liab", "termin", "docum", "custom", "marit", "agric",
    "agricult", "infest", "univers", "prof", "profess", "housek", "bronch",
    "pulmon", "cardiov", "cardiovasc", "arrhythm", "infarct", "embol", "pneum",
    "pneumon", "pediat", "pediatr", "ergon", "ergonom", "obfusc", "unauth",
    "unauthor", "contam", "contamin", "degrad", "catast", "catastr", "immed",
}

GRAMMAR_GLUE = {
    "a", "an", "the", "this", "that", "these", "those", "i",
    "is", "are", "was", "were", "be", "been", "being",
    "it", "its", "to", "of", "in", "for", "with", "on", "at", "from", "by",
    "into", "onto", "about", "through", "under", "over",
    "and", "or", "as", "than",
}

QUESTION_STOPWORDS = GRAMMAR_GLUE | {
    "what", "which", "who", "where", "when", "why", "how", "whose", "whom",
    "did", "does", "do", "has", "have", "had", "will", "would", "should",
    "could", "can", "may", "might", "must", "shall", "need", "needs",
    "one", "word", "answer", "question", "type", "kind", "level", "specific",
    "immediate", "primary", "main", "current", "right", "now", "there", "here",
    "handle", "handles", "handled", "take", "taken", "took", "make", "made",
    "cause", "caused", "affect", "affected", "receive", "received", "ask",
    "asked", "meet", "met", "target", "targeted", "cover", "covered",
}

# Domain-Disjoint Anchor Prompt (Astronomy & Baking)
GUIDED_ANCHOR_PROMPT = (
    "Task: Answer each question about the situation in a single lowercase word.\n\n"
    "Situation: The orbital observatory detected an intense gamma ray burst from a collapsing hypergiant.\n"
    "Question: What astronomical instrument made the discovery? One-word answer: telescope.\n"
    "Question: What celestial object underwent collapse? One-word answer: star.\n"
    "Question: What electromagnetic emission was recorded? One-word answer: gamma.\n"
    "Question: Did the collapsing star remain stable? One-word answer: no.\n\n"
    "Situation: The artisan pastry chef kneaded flour, chilled butter, and water to prepare a fruit tart.\n"
    "Question: What baked pastry is being crafted? One-word answer: tart.\n"
    "Question: What dairy ingredient was incorporated? One-word answer: butter.\n"
    "Question: What culinary specialist prepared the dough? One-word answer: chef.\n"
    "Question: Did the chef use chilled butter? One-word answer: yes.\n\n"
)
