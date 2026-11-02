"""Error analysis: tag the hard cases and measure each classifier on them.

Tags are simple surface patterns, so they select candidates, not verified
phenomena. The report reads examples from each tag by hand before saying what
they show.

- technical: the text uses software vocabulary that general-purpose lexicons
  score as negative ("kill", "fatal", "fails", "crash", ...). On items whose
  gold label is not negative, predicting negative is the classic lexicon error
  (Jongeling et al. 2017; Novielli et al. 2018).
- politeness: thanks, please, sorry and similar. The gold sets label these
  differently: Jira tends to count thanks for a patch as positive and apologies
  as negative, Stack Overflow and GitHub treat routine politeness as neutral
  (Novielli et al. 2020).
- sarcasm candidate: gold negative, yet VADER's compound score is clearly
  positive (>= 0.5): positive words in a negative message, which is where
  sarcasm, irony and mixed messages show up.
"""

from __future__ import annotations

import re

TECHNICAL_TERMS = (
    "kill", "killed", "kills", "killing", "fail", "fails", "failed", "failing", "failure", "failures",
    "error", "errors", "crash", "dead", "die", "fatal", "broken", "wrong", "problem", "problems",
    "reject", "rejected", "destroy", "panic", "conflict", "conflicts", "warning", "warnings", "leak",
    "block", "blocked", "attack", "threat", "dump",
)
TECHNICAL = re.compile(r"\b(" + "|".join(TECHNICAL_TERMS) + r")\b", re.IGNORECASE)
POLITENESS = re.compile(r"\b(thanks?|thank you|thx|please|pls|sorry|apolog\w*|appreciated?|cheers)\b", re.IGNORECASE)
SARCASM_COMPOUND = 0.5

TAGS = {
    "technical": "Technical vocabulary a general lexicon scores as negative, gold not negative",
    "politeness": "Thanks, please, sorry and similar",
    "sarcasm": "Sarcasm candidates: gold negative, VADER compound >= 0.5",
}


def tags_for(text: str, gold: str, vader_compound: float | None = None) -> list[str]:
    tags = []
    if gold != "negative" and TECHNICAL.search(text):
        tags.append("technical")
    if POLITENESS.search(text):
        tags.append("politeness")
    if gold == "negative" and vader_compound is not None and vader_compound >= SARCASM_COMPOUND:
        tags.append("sarcasm")
    return tags


MENTION = re.compile(r"(?<![\w.])@[A-Za-z0-9_][A-Za-z0-9_.-]*")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")


def mask(text: str) -> str:
    """Hide @handles and e-mail addresses before texts are shown in the web demo."""
    return MENTION.sub("@user", EMAIL.sub("<email>", text))
