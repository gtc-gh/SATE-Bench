"""
Prompts configuration for Single-Turn Active/Passive VQA GRPO Training with Thinking Process.
"""

GRPO_SYSTEM_PROMPT = """You are an expert in visual reasoning and 3D spatial understanding. \
You are analyzing a single camera viewpoint image of a scene to solve a multiple-choice question.

Your task is to carefully analyze the visual evidence, answer the question, and make a decision:
1. ANSWER: If the visual evidence in this image is clear, unambiguous, and sufficient to confidently answer the question, choose the correct factual option.
2. EXPLORE: If the enquired target is out of field of view or invisible, or if the visual evidence is occluded, insufficient, or requires observing the scene from other angles, choose the exploration option (e.g. "Move and explore more views.").

OUTPUT FORMAT REQUIREMENT:
You must first write your step-by-step reasoning between <think> and </think> tags.
Then, provide your final chosen option letter formatted strictly within \\boxed{LETTER} (for example, \\boxed{A})."""


def format_options_text(options: dict) -> str:
    """Formats the options dictionary into lines like 'A) ... \nB) ...'."""
    keys = sorted(options.keys())
    return "\n".join([f"{k}) {options[k]}" for k in keys])


def build_prompt(sample: dict) -> list:
    """
    Constructs the multimodal chat message template for a dataset sample.
    Compatible with PyArrow schema inference and Hugging Face multimodal chat templates
    (Qwen2.5-VL, Gemma-3, etc.).
    """
    options = sample.get("options", {})
    options_text = format_options_text(options)
    allowed_letters = ", ".join(sorted(options.keys()))

    user_text = (
        "Please carefully analyze the provided image and answer the question. "
        "You MUST format your output exactly as shown in the examples below.\n\n"
        "--- Example 1 ---\n"
        "Question:\n"
        "Is there a person wearing a blue shirt behind the wall?\n\n"
        "Options:\n"
        "A) Yes.\n"
        "B) No.\n"
        "C) Move and explore more views.\n\n"
        "Output:\n"
        "<think>\n"
        "Looking at the current view, my line of sight is severely occluded by the wall. "
        "I cannot see what is behind it. Therefore, I do not have sufficient information to answer "
        "the question and must explore adjacent viewpoints.\n"
        "</think>\n"
        "\\boxed{C}\n"
        "--- End of Example 1 ---\n\n"
        "--- Example 2 ---\n"
        "Question:\n"
        "Is the person wearing a red hat sitting on the sofa?\n\n"
        "Options:\n"
        "A) Yes.\n"
        "B) No.\n"
        "C) Move and explore more views.\n\n"
        "Output:\n"
        "<think>\n"
        "I have carefully scanned the current view. While I can see the sofa, the person wearing a red hat is entirely invisible and does not exist in this frame. Since they might be located just outside the current camera angle, I cannot definitively answer without checking other perspectives.\n"
        "</think>\n"
        "\\boxed{C}\n"
        "--- End of Example 2 ---\n\n"
        "Now, answer the following question:\n\n"
        f"Question:\n{sample.get('question', '')}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Output:\n"
        f"First output your thinking process inside <think>...</think>, and then output your final answer as \\boxed{{LETTER}} choosing from [{allowed_letters}]."
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": GRPO_SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]}
    ]
    return messages
