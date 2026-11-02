"""Prompts for zero-shot and few-shot LLM classification.

zs-basic, zs-guided and fs-guided were written before any model was run.
fs-basic was added after the first dev round, which showed the guided rules
conflicting with Jira's labelling conventions (thanks for a patch is positive
there, an apology negative). Every variant was compared on the dev splits only
(docs/benchmark.md, "Prompt selection on dev"). Selection rule, fixed before
test was scored: per model, the zero-shot and the few-shot variant with the
highest mean dev macro-F1 across the three datasets. Test was scored once.

- zs-basic: a one-line instruction, as a naive user would write it.
- zs-guided: the gold standards' annotation scheme (polarity from Shaver et
  al.'s emotion framework, as described by Calefato et al. 2018 and Novielli et
  al. 2020), plus rules for the known hard cases: technical vocabulary,
  politeness, sarcasm and mixed messages.
- fs-guided@<dataset>: zs-guided plus nine labelled examples, three per class,
  drawn at random (fixed seed) from that dataset's fit split. The examples come
  from the fit split, never from dev or test.
- fs-basic@<dataset>: zs-basic plus the same nine examples, so the examples
  alone carry the dataset's conventions.
"""

from __future__ import annotations

import random
from functools import cache

from .data import LABELS, load_split

SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": list(LABELS)}},
    "required": ["label"],
    "additionalProperties": False,
}
SCHEMA_NAME = "sentiment_polarity"

MAX_CHARS = 2000  # a handful of Jira comments are longer; they are cut here and the cut is marked
SHOTS_PER_CLASS = 3
SHOT_MAX_CHARS = 300
SHOT_SEED = 7

BASIC = ("You are a sentiment classifier for software-engineering text. Classify the overall sentiment "
         "polarity of the message as positive, neutral or negative. Answer in JSON.")

GUIDED = """You label the sentiment polarity of one message written by a software developer: an issue comment, a pull-request or commit comment, or a Stack Overflow question, answer or comment.

Label the emotion the writer expresses, following the annotation scheme of the gold standard (polarity derived from Shaver et al.'s emotion framework):
- positive: the writer expresses a positive emotion such as joy or love: enthusiasm, satisfaction, appreciation, excitement, relief.
- negative: the writer expresses a negative emotion such as anger, sadness or fear: frustration, annoyance, disappointment, worry, discouragement.
- neutral: no emotion is expressed. Most technical statements, questions, status reports and requests are neutral.

Rules:
1. Words that describe software behaviour are not emotions. "Kill the process", "fatal error", "this test fails", "abort the build" and "a bug in the parser" are neutral unless the writer also shows a feeling about it.
2. Routine politeness ("please", "thanks", "sorry for the delay") does not decide the label on its own; label the emotion behind it. Emphatic gratitude or praise ("Thanks a lot, this is awesome!") is positive.
3. Sarcasm and irony carry the opposite of their literal polarity: "Great, another broken build." is negative.
4. If a message mixes emotions, label the one that dominates.

Return JSON with the label only."""

PROMPT_IDS = ("zs-basic", "zs-guided", "fs-basic", "fs-guided")


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " [...]"


@cache
def shots(dataset: str) -> tuple[dict, ...]:
    """Three short examples per class from the dataset's fit split, in a fixed shuffled order."""
    rng = random.Random(SHOT_SEED)
    fit = [it for it in load_split(dataset, "fit") if len(it["text"]) <= SHOT_MAX_CHARS]
    chosen = []
    for label in LABELS:
        group = [it for it in fit if it["label"] == label]
        chosen += rng.sample(group, SHOTS_PER_CLASS)
    rng.shuffle(chosen)
    return tuple(chosen)


def parse_prompt_id(prompt_id: str) -> tuple[str, str | None]:
    """'fs-guided@jira' -> ('fs-guided', 'jira')."""
    base, _, source = prompt_id.partition("@")
    if base not in PROMPT_IDS:
        raise ValueError(f"unknown prompt {base!r}; expected one of {PROMPT_IDS}")
    if base.startswith("fs-") and not source:
        raise ValueError(f"few-shot prompt needs a source dataset, e.g. {base}@github")
    return base, source or None


def system_prompt(prompt_id: str) -> str:
    base, source = parse_prompt_id(prompt_id)
    if base == "zs-basic":
        return BASIC
    if base == "zs-guided":
        return GUIDED
    examples = "\n\n".join(f'Message: """{it["text"]}"""\nLabel: {it["label"]}' for it in shots(source))
    instructions = GUIDED if base == "fs-guided" else BASIC
    return f"{instructions}\n\nLabelled examples from the gold standard:\n\n{examples}"


def user_prompt(text: str) -> str:
    return f'Message: """{_clip(text, MAX_CHARS)}"""'
