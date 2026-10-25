"""Prompt templates used in the SATE-Bench evaluation (see Appendix G of the paper)."""

# =============================================================================
# Passive setting (single-turn, bag of frames)
# =============================================================================
SYSTEM_PROMPT_PASSIVE = """You are an expert in multi-view humans and objects understanding. \
You will be provided with several views of a scene. Carefully analyze the images based on the user's question, \
then provide the final answer as a single letter corresponding to the correct option. \
You must answer strictly with one single letter from [A, B, C, D, E]. Do not add any explanation or extra text. Directly answer with a single letter!"""

USER_TEMPLATE_PASSIVE_5 = """You are given multiple synchronized views of the same scene.

Question:
{question}

Options:
A) {optA}
B) {optB}
C) {optC}
D) {optD}
E) {optE}

Output:
Answer with exactly ONE letter from [A, B, C, D, E], e.g., just "C". No explanation. Only one letter."""

USER_TEMPLATE_PASSIVE_2 = """You are given multiple synchronized views of the same scene.

Question:
{question}

Options:
A) {optA}
B) {optB}

Output:
Answer with exactly ONE letter from [A, B], e.g., just "B". No explanation. Only one letter."""


# =============================================================================
# Active setting (multi-turn view exploration)
# =============================================================================
SYSTEM_PROMPT_ACTIVE = """You are an expert in multi-view humans and objects understanding. \
You are currently viewing 1 out of {num_views} possible camera viewpoints of a single scene. \
These viewpoints are uniformly distributed in a circle around the scene and sorted in clockwise order.

Your task is to answer the user's multiple-choice question by carefully analyzing the provided image. \
You must make a strict decision between answering the question or exploring the scene further:

1. ANSWER: If the current view provides sufficient information to confidently answer the question, output the correct option.
2. EXPLORE: If the current view lacks crucial information, is heavily occluded, or \
if the question requires full-scene context that this single angle cannot provide, you must request a new viewpoint.\x20
   - Option 'X': Move to the adjacent left (clockwise) viewpoint.
   - Option 'Y': Move to the adjacent right (counter-clockwise) viewpoint.

CRITICAL CONSTRAINT:
Your final output must be exactly ONE single letter from the set [A, B, C, D, E, X, Y].\x20
Do not include any explanations, formatting, punctuation, or extra text. Directly output the single letter."""

USER_TEMPLATE_ACTIVE_5 = """You are given 1 view or multiple views of the scene.

Question:
{question}

Options:
A) {optA}
B) {optB}
C) {optC}
D) {optD}
E) {optE}

Exploration Options:
X) {optX}
Y) {optY}

Output:
Answer with exactly ONE letter from [A, B, C, D, E, X, Y], e.g., just "C". No explanation. Only one letter."""

USER_TEMPLATE_ACTIVE_2 = """You are given 1 view or multiple views of the scene.

Question:
{question}

Options:
A) {optA}
B) {optB}

Exploration Options:
X) {optX}
Y) {optY}

Output:
Answer with exactly ONE letter from [A, B, X, Y], e.g., just "B". No explanation. Only one letter."""

OPT_X_TEXT = "Move to adjacent left (clockwise) viewpoint"
OPT_Y_TEXT = "Move to adjacent right (counter-clockwise) viewpoint"


# =============================================================================
# Answer extractor (fallback when rule-based parsing fails)
# =============================================================================
EXTRACTOR_SYSTEM_PROMPT = """You are an expert in answer extraction. Your task is to identify the final selected option from the provided model output.\x20
If the prediction explicitly chooses option A, B, C, D, or E, return the corresponding letter.\x20
If the prediction does not specify a choice or indicates that none of the options are correct, return None.\x20
Output ONLY the single letter (A, B, C, D, or E) or None. Do not provide explanations!"""

# Active setting: the model may also choose an exploration move (X / Y).
EXTRACTOR_SYSTEM_PROMPT_ACTIVE = """You are an expert in answer extraction. Your task is to identify the final selected option from the provided model output.\x20
If the prediction explicitly chooses option A, B, C, D, or E, return the corresponding letter.\x20
If the prediction chooses to move to another viewpoint, return X for the adjacent left (clockwise) viewpoint or Y for the adjacent right (counter-clockwise) viewpoint.\x20
If the prediction does not specify a choice or indicates that none of the options are correct, return None.\x20
Output ONLY the single letter (A, B, C, D, E, X, or Y) or None. Do not provide explanations!"""


# =============================================================================
# Prompt builders
# =============================================================================
def _option_kwargs(sample: dict) -> dict:
    options = sample["options"]
    return {
        "question": sample["question"],
        "optA": options.get("A", ""), "optB": options.get("B", ""),
        "optC": options.get("C", ""), "optD": options.get("D", ""), "optE": options.get("E", ""),
    }


def build_passive_prompt(sample: dict) -> str:
    template = USER_TEMPLATE_PASSIVE_2 if len(sample["options"]) == 2 else USER_TEMPLATE_PASSIVE_5
    return template.format(**_option_kwargs(sample))


def build_active_prompts(sample: dict) -> tuple:
    """Returns (system_prompt, user_prompt) for one Active-setting sample."""
    template = USER_TEMPLATE_ACTIVE_2 if len(sample["options"]) == 2 else USER_TEMPLATE_ACTIVE_5
    user_prompt = template.format(optX=OPT_X_TEXT, optY=OPT_Y_TEXT, **_option_kwargs(sample))
    return SYSTEM_PROMPT_ACTIVE.format(num_views=sample["num_views"]), user_prompt
