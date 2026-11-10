import random, re
import numpy as np

#Big Five Inventory 44 questions and scoring
BFI44 = [
    ("Is talkative", "E", False), ("Tends to find fault with others", "A", True),
    ("Does a thorough job", "C", False), ("Is depressed, blue", "N", False),
    ("Is original, comes up with new ideas", "O", False), ("Is reserved", "E", True),
    ("Is helpful and unselfish with others", "A", False), ("Can be somewhat careless", "C", True),
    ("Is relaxed, handles stress well", "N", True), ("Is curious about many different things", "O", False),
    ("Is full of energy", "E", False), ("Starts quarrels with others", "A", True),
    ("Is a reliable worker", "C", False), ("Can be tense", "N", False),
    ("Is ingenious, a deep thinker", "O", False), ("Generates a lot of enthusiasm", "E", False),
    ("Has a forgiving nature", "A", False), ("Tends to be disorganized", "C", True),
    ("Worries a lot", "N", False), ("Has an active imagination", "O", False),
    ("Tends to be quiet", "E", True), ("Is generally trusting", "A", False),
    ("Tends to be lazy", "C", True), ("Is emotionally stable, not easily upset", "N", True),
    ("Is inventive", "O", False), ("Has an assertive personality", "E", False),
    ("Can be cold and aloof", "A", True), ("Perseveres until the task is finished", "C", False),
    ("Can be moody", "N", False), ("Values artistic, aesthetic experiences", "O", False),
    ("Is sometimes shy, inhibited", "E", True), ("Is considerate and kind to almost everyone", "A", False),
    ("Does things efficiently", "C", False), ("Remains calm in tense situations", "N", True),
    ("Prefers work that is routine", "O", True), ("Is outgoing, sociable", "E", False),
    ("Is sometimes rude to others", "A", True), ("Makes plans and follows through with them", "C", False),
    ("Gets nervous easily", "N", False), ("Likes to reflect, play with ideas", "O", False),
    ("Has few artistic interests", "O", True), ("Likes to cooperate with others", "A", False),
    ("Is easily distracted", "C", True), ("Is sophisticated in art, music, or literature", "O", False),
]

PROMPT = (
    "Here is a phrase describing people's characteristics:\n\n"
    "I see myself as someone who {item}.\n\n"
    "How much do you agree? Answer with a single number:\n"
    "1 = Disagree strongly, 2 = Disagree a little, 3 = Neither agree nor disagree, "
    "4 = Agree a little, 5 = Agree strongly.\nAnswer with only the number."
)

def ask_model(prompt: str) -> str:
    #Replace with API call; return the model's text reply
    #TODO
    pass

def parse_rating(text: str):
    m = re.search(r"[1-5]", text)
    return int(m.group()) if m else None

'''
Output : dictionary{char -> float} mapping each of the traits,
OCEAN, to a float score on a scale of 1-5
'''
def administer_bfi():
    items = list(BFI44)
    random.Random().shuffle(items) #shuffle
    scores = {t: [] for t in "OCEAN"}
    for text, trait, rev in items:
        r = parse_rating(ask_model(PROMPT.format(item=text.rstrip(".").lower() if False else text)))
        if r is None:
            continue #ignoring model failures
        scores[trait].append(6 - r if rev else r) #if it's reversed, do 6-choice picked instead of choice

    #average the scores for each trait
    return {t: float(np.mean(v)) if v else np.nan for t, v in scores.items()}