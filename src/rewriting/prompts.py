"""Chat prompts for each pipeline stage.

Only `rewrite_messages` and `no_draft_messages` receive a target. Extraction is
blind to it, so a decision can change only if the text itself changes. No stage
sees an item's endorsement key.
"""
from .traits import describe_target

SPEAKER = ("You are writing as an ordinary person in the first person. "
           "Never mention being an AI.")

DRAFT_TASK = (
    "In 2-3 first-person sentences, say what you would do and why. Commit to one "
    "course of action. Do not use the words yes or no, and do not hedge."
)
BFI_DRAFT_TASK = (
    "In 2-3 first-person sentences, say how well this describes you and why. "
    "Commit to a clear position and do not use any numbers."
)

REWRITE_RULES = {
    "free": ("Rewrite the statement so it sounds like this person. You may change "
             "their tone, reasoning, and what they decide to do, if this person "
             "would act differently. Keep the facts of the situation unchanged."),
    "locked": ("Rewrite the statement so it sounds like this person. Change only "
               "tone and word choice. The speaker must still decide exactly the "
               "same thing for the same practical outcome."),
}
PARAPHRASE_RULE = ("Paraphrase the statement in different words. Keep the meaning, "
                   "tone, and decision exactly the same.")
OUTPUT_RULE = ("Write 2-3 first-person sentences. Do not use the words yes or no. "
               "Output only the statement, with no preamble.")


def _chat(system, user):
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _target_block(trait_code, level):
    return (f"{describe_target(trait_code, level)} "
            "All other personality traits are neutral (3 out of 5).")


def draft_messages(context):
    task = BFI_DRAFT_TASK if context.kind == "bfi" else DRAFT_TASK
    return _chat(SPEAKER, f"{context.prompt}\n\n{task}")


def rewrite_messages(context, draft, trait_code, level, mode):
    if mode not in REWRITE_RULES:
        raise ValueError(f"Unknown rewrite mode: {mode!r}")
    return _chat(SPEAKER, f"{context.prompt}\n\nOriginal statement:\n{draft}\n\n"
                          f"{_target_block(trait_code, level)}\n\n"
                          f"{REWRITE_RULES[mode]} {OUTPUT_RULE}")


def paraphrase_messages(context, draft):
    return _chat(SPEAKER, f"{context.prompt}\n\nOriginal statement:\n{draft}\n\n"
                          f"{PARAPHRASE_RULE} {OUTPUT_RULE}")


def no_draft_messages(context, trait_code, level):
    """The free-mode request with the draft removed, to show what the draft adds."""
    task = BFI_DRAFT_TASK if context.kind == "bfi" else DRAFT_TASK
    return _chat(SPEAKER, f"{context.prompt}\n\n{_target_block(trait_code, level)}\n\n"
                          f"Write as this person. {task}")


def extract_messages(context, statement):
    system = "You read statements and report what the speaker said. You never give your own opinion."
    if context.kind == "bfi":
        ask = ("Based only on the statement, how much does the speaker agree with the "
               "description? 1 = Disagree strongly, 2 = Disagree a little, 3 = Neither "
               "agree nor disagree, 4 = Agree a little, 5 = Agree strongly. Reply with "
               "only the number, or Unclear if the statement takes no position.")
    elif context.kind == "yes_no":
        # Name the behavior: "do it" pointed back several lines and was misread.
        ask = (f"Based only on the statement, did the speaker decide to do this: "
               f"'{context.item['behavior']}'? Answer Yes if they say they will do it, "
               "No if they say they will not or will do something else instead, and "
               "Unclear only if they leave it open. Reply with exactly one word: "
               "Yes, No, or Unclear.")
    else:
        options = " / ".join(context.displayed_options)
        if "hesitant" in context.prompt.lower():
            # These items ask which option the speaker would avoid; the extractor
            # reported the one they picked. A general version of this instruction
            # made it pick the rejected option on ordinary items, so it is scoped.
            ask = ("This question asks which option the speaker is more hesitant about. "
                   "Based only on the statement, which option does the speaker hesitate "
                   "over or avoid? Give that option, not the one they would choose. "
                   f"Options: {options}. Reply with the exact option text, or Unclear.")
        else:
            ask = ("Based only on the statement, which option did the speaker choose? "
                   f"Options: {options}. Reply with the exact option text, or Unclear.")
    return _chat(system, f"{context.prompt}\n\nStatement:\n{statement}\n\n{ask}")
