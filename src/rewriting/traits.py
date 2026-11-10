"""Trait text shared by conditioning methods.

Descriptions are generic trait definitions. They deliberately avoid the
BehaviorBench proxies (for example risk-taking for Openness), so that a result
cannot come from naming the benchmark's behavior in the instruction.
"""

TRAITS = {
    "O": {
        "name": "Openness",
        # Development tuning: an earlier ending, "and quick to question convention",
        # made rewrites question the draft's own decision and hedge at high levels.
        "high": "imaginative, curious, and drawn to unfamiliar ideas and experiences",
        "low": "practical, attached to familiar ways of doing things, and uninterested in abstract ideas",
    },
    "C": {
        "name": "Conscientiousness",
        "high": "disciplined, orderly, and careful to plan ahead and honor commitments",
        "low": "spontaneous, loose about rules and deadlines, and inclined to act on the mood of the moment",
    },
    "E": {
        "name": "Extraversion",
        "high": "energized by people, expressive, and eager to be noticed and to join in",
        "low": "energized by solitude, private, and inclined to stay in the background",
    },
    "A": {
        "name": "Agreeableness",
        "high": "warm, cooperative, and inclined to put other people's needs first",
        "low": "blunt, skeptical of others' motives, and inclined to put their own interests first",
    },
    "N": {
        "name": "Neuroticism",
        "high": "anxious, emotionally reactive, and prone to dwelling on what could go wrong",
        "low": "even-tempered, emotionally steady, and rarely unsettled by setbacks",
    },
}

LEVELS = (1, 2, 3, 4, 5)
LEVEL_INTENSITY = {1: "very low", 2: "somewhat low", 3: "neutral", 4: "somewhat high", 5: "very high"}


def describe_target(trait_code, level):
    """Return the score and its interpretation for one trait at one level."""
    if level not in LEVEL_INTENSITY:
        raise ValueError(f"Target level must be one of {LEVELS}, got {level!r}")
    trait = TRAITS[trait_code]
    name, intensity = trait["name"], LEVEL_INTENSITY[level]
    header = f"{name} score: {level} on a scale from 1 to 5 ({intensity})."
    if level == 3:
        return (f"{header} This person is neither {trait['high']} "
                f"nor {trait['low']}; they sit in the middle of this trait.")
    pole = trait["high"] if level > 3 else trait["low"]
    degree = "strongly" if level in (1, 5) else "moderately"
    return f"{header} This person is {degree} {pole}."
