"""
Build the conversation corpus that teaches the model to answer, not just continue.

WHY THIS FILE EXISTS
--------------------
"hi -> hello" cannot be added in model.py. The model is a next-character
predictor: it only ever reproduces patterns that existed in its training text.
wiki_medical_terms contains no dialogue at all, so the model has never seen a
greeting followed by a reply. You teach it by example, then fine-tune.

TWO DESIGN CHOICES THAT MATTER
------------------------------
1. The turn markers are "User:" and "Doctor:" — plain ASCII. Every one of
   those characters is ALREADY in v2's 249-character vocabulary, so the
   tokenizer does not change and the v2 weights stay loadable. If we invented
   tokens like <|user|> we would have to grow the embedding table and could no
   longer warm-start from v2.
2. Replies are deliberately deflecting on anything clinical. A 4.86M-parameter
   character model must never look like it is giving medical advice, so the
   safe answers are the ones we train in on purpose.

Run:  python chat_data.py   ->  writes chat_data.txt
"""
import itertools
import random

random.seed(1337)

# Each entry: (list of ways a user might say it, list of acceptable replies).
# Multiple phrasings per intent is what lets the model generalise instead of
# memorising one exact string.
INTENTS = [
    (   # greetings — the thing you actually asked for
        ["hi", "hello", "hey", "hi there", "hello there", "hey there",
         "good morning", "good afternoon", "good evening", "hi doctor",
         "hello doctor", "hii", "helo", "yo"],
        ["Hello. How can I help you today?",
         "Hello. What would you like to know about?",
         "Hi. What can I help you with?",
         "Hello. Ask me about a medical term and I will explain it.",
         "Hi there. How can I help?"],
    ),
    (   # who/what are you
        ["who are you", "what are you", "what is your name", "introduce yourself",
         "what can you do", "what do you do", "are you a doctor",
         "are you human", "are you real"],
        ["I am a small research language model trained on medical reference text.",
         "I am a character level language model built for research, not a clinician.",
         "I am an experimental medical text model. I am not a doctor.",
         "I am a research model trained on medical terminology. I cannot give advice."],
    ),
    (   # thanks / goodbye
        ["thanks", "thank you", "thanks a lot", "thank you so much", "ok thanks",
         "bye", "goodbye", "see you", "that is all", "ok bye"],
        ["You are welcome.", "Happy to help.", "Goodbye. Take care.",
         "You are welcome. Take care.", "Glad to help."],
    ),
    (   # the safety-critical intent: never answer clinically
        ["i have a headache", "i have chest pain", "i feel sick", "i have a fever",
         "my stomach hurts", "what medicine should i take", "what should i do",
         "am i sick", "do i have diabetes", "is this serious", "can you diagnose me",
         "what is wrong with me", "should i go to hospital", "how much should i take"],
        ["I cannot give medical advice. Please speak to a qualified clinician.",
         "I am not able to diagnose. Please see a doctor about this.",
         "Please consult a healthcare professional. I cannot advise on your case.",
         "I cannot assess symptoms. A clinician should evaluate this."],
    ),
    (   # confusion / repair
        ["i do not understand", "what", "sorry", "can you repeat that",
         "explain again", "that makes no sense"],
        ["Let me put it another way. What would you like explained?",
         "I can try again. Which part should I clarify?",
         "Sorry. Ask me about a specific medical term."],
    ),
]

# Definition-style turns: this is the one thing the base model genuinely knows,
# so we teach it to deliver that knowledge inside the dialogue format rather
# than inventing new facts. The reply is left OPEN so fine-tuning learns to
# continue with real wiki-style prose from the pretrained weights.
TERMS = ["diabetes", "pneumonia", "asthma", "hypertension", "anaemia",
         "arthritis", "bronchitis", "cirrhosis", "dermatitis", "epilepsy",
         "gastritis", "hepatitis", "influenza", "leukaemia", "meningitis",
         "migraine", "nephritis", "osteoporosis", "psoriasis", "sepsis",
         "sinusitis", "tuberculosis", "ulcer", "vasculitis", "glaucoma"]
TERM_QUESTIONS = ["what is {t}", "tell me about {t}", "explain {t}",
                  "what does {t} mean", "define {t}", "what causes {t}"]


def build():
    """Writes chat_data.txt (train) and chat_holdout.txt (honest eval).

    THE SPLIT MATTERS MORE THAN THE SIZE.  A random split over dialogues is
    worthless here: every dialogue draws from the same small pool of replies,
    so train and validation end up sharing identical target strings and the
    val loss measures memorisation instead of generalisation.

    So we hold out whole PHRASINGS instead — user wordings the model will
    never have seen. If it still answers those with the right intent, it
    generalised. If loss is low on train but high on holdout, it memorised.
    """
    convos, holdout = [], []

    for prompts, replies in INTENTS:
        # last two phrasings of each intent are held back entirely
        train_p, held_p = prompts[:-2], prompts[-2:]
        # every phrasing paired with every reply, so no single surface form
        # gets locked to a single answer
        for p, r in itertools.product(train_p, replies):
            convos.append(f"User: {p}\nDoctor: {r}\n")
        for p, r in itertools.product(held_p, replies):
            holdout.append(f"User: {p}\nDoctor: {r}\n")

    for t, q in itertools.product(TERMS, TERM_QUESTIONS):
        convos.append(f"User: {q.format(t=t)}\nDoctor: {t.capitalize()} is")

    # Multi-turn examples: without these the model learns to stop after one
    # exchange and never handles a follow-up.
    greet = INTENTS[0]
    safety = INTENTS[3]
    for _ in range(300):
        g = random.choice(greet[0])
        gr = random.choice(greet[1])
        s = random.choice(safety[0])
        sr = random.choice(safety[1])
        convos.append(f"User: {g}\nDoctor: {gr}\nUser: {s}\nDoctor: {sr}\n")

    random.shuffle(convos)
    text = "\n".join(convos)
    with open("chat_data.txt", "w") as f:
        f.write(text)
    with open("chat_holdout.txt", "w") as f:
        f.write("\n".join(holdout))
    print(f"chat_data.txt    : {len(convos)} dialogues, {len(text):,} chars")
    print(f"chat_holdout.txt : {len(holdout)} dialogues on UNSEEN phrasings")

    # The number that actually predicts memorisation: how many DISTINCT
    # sentences the model has to reproduce. Count per line — counting per
    # dialogue would include multi-turn tails and inflate the figure.
    lines = [l for l in text.split("\n") if l.startswith("Doctor: ")]
    print(f"Doctor lines: {len(lines)}, unique: {len(set(lines))}")
    print(f"  ^ THIS is the memorisation ceiling. The model can only ever "
          f"recite these {len(set(lines))} sentences.")
    print("  Raise it by adding replies/intents above — extra training steps "
          "cannot create variety that is not in the data.")
    print("\nNOTE: chat_holdout.txt varies only the USER phrasing; the Doctor "
          "replies are the same sentences.\n  So chat_val measures 'does it map "
          "an unseen wording to the right reply', which is the part that CAN "
          "generalise.\n  It does not measure novel phrasing — a canned-reply "
          "corpus cannot test that.")
    return text


if __name__ == "__main__":
    build()
