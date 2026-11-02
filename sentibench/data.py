"""The three SE sentiment gold sets: download, clean, split and sample.

Every item is {"id", "text", "label"} with label in positive / neutral / negative.

Splits:
- Jira and Stack Overflow ship with an official train/test partition (the one
  distributed with Senti4SD and EMTk); it is used as is.
- GitHub has no official partition. Following Novielli et al. (MSR 2020), it is
  split 70/30 with stratified sampling, here with a fixed seed.
- A dev set (20 % of train, stratified, fixed seed) is held out of every train
  split. Prompts and hyperparameters are chosen on dev; supervised models are
  then refitted on the full train split and test is scored once.

Samples: the API models are too expensive to run on every test item, so each
split also gets a fixed "stratified prefix order": any prefix of that order has
roughly the split's class balance, and smaller samples are prefixes of larger
ones (the variance sample is inside the LLM sample, which is inside test).
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import urllib.request
from collections import Counter
from pathlib import Path

LABELS = ("negative", "neutral", "positive")
DATASETS = ("github", "jira", "so")
NAMES = {"github": "GitHub", "jira": "Jira", "so": "Stack Overflow"}
SEED = 2026
DEV_FRACTION = 0.2
GITHUB_TEST_FRACTION = 0.3

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

_EMTK = "https://raw.githubusercontent.com/collab-uniba/EMTK_datasets/2592accde5f286699e62dfb284eda161604465b4"
SOURCES = {
    "github/github_gold.csv": ("https://ndownloader.figshare.com/files/21001260", "3166bb1f802b4558723e5899928cd983"),
    "jira/train_jira.csv": (f"{_EMTK}/jira/emotions/train_jira.csv", "c24db6c96a1e795bffdc9079d93b08ec"),
    "jira/test_jira.csv": (f"{_EMTK}/jira/emotions/test_jira.csv", "d988d95c0356f5c8ebe1d9fe65bdeb53"),
    "so/train3098itemPOLARITY.csv": (f"{_EMTK}/so/polarity/train3098itemPOLARITY.csv", "93e9ff6e08ea86ce5b4366ea0401c127"),
    "so/test1326itemPOLARITY.csv": (f"{_EMTK}/so/polarity/test1326itemPOLARITY.csv", "f30118b2fa02d9528eee867737fa8113"),
}


def fetch(raw_dir: Path = RAW_DIR) -> list[Path]:
    """Download the raw gold sets (about 2.4 MB in total) unless already present."""
    paths = []
    for rel, (url, md5) in SOURCES.items():
        path = raw_dir / rel
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=60) as r:
                path.write_bytes(r.read())
        if md5 and hashlib.md5(path.read_bytes()).hexdigest() != md5:
            raise ValueError(f"{path} does not match the published checksum")
        paths.append(path)
    return paths


def _read_csv(path: Path, header: bool) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter=";"))
    return rows[1:] if header else rows


def clean_github_text(text: str) -> str:
    """The GitHub CSV wraps each comment in an extra quote level: almost every text
    ends in a stray '"' and inner quotes are doubled. Undo both."""
    if text.endswith('"') and text.count('"') % 2 == 1:
        text = text[:-1]
    return text.replace('""', '"').strip()


def _items(rows: list[list[str]], prefix: str, clean=None) -> list[dict]:
    items = []
    for i, (rid, label, text) in enumerate(rows):
        text = clean(text) if clean else text.strip()
        if not text:
            continue  # one empty Jira test row
        assert label in LABELS, label
        # Jira ids are not unique (a few comments appear twice), so the row number is part of the id.
        items.append({"id": f"{prefix}-{i:05d}-{rid}", "text": text, "label": label})
    return items


def stratified_split(items: list[dict], fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """(rest, held_out), with held_out about `fraction` of every class."""
    rng = random.Random(seed)
    rest, held = [], []
    for label in LABELS:
        group = [it for it in items if it["label"] == label]
        rng.shuffle(group)
        n = round(len(group) * fraction)
        held += group[:n]
        rest += group[n:]
    key = {it["id"]: k for k, it in enumerate(items)}
    return sorted(rest, key=lambda it: key[it["id"]]), sorted(held, key=lambda it: key[it["id"]])


def stratified_order(items: list[dict], seed: int) -> list[str]:
    """Ids in an order where every prefix keeps roughly the class balance.
    Within each class the order is a seeded shuffle; classes are interleaved by
    relative rank (k + 0.5) / class size."""
    rng = random.Random(seed)
    keyed = []
    for label in LABELS:
        group = [it["id"] for it in items if it["label"] == label]
        rng.shuffle(group)
        keyed += [((k + 0.5) / len(group), rng.random(), i) for k, i in enumerate(group)]
    return [i for *_, i in sorted(keyed)]


def prepare(raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR) -> dict:
    github = _items(_read_csv(raw_dir / "github/github_gold.csv", True), "gh", clean_github_text)
    gh_train, gh_test = stratified_split(github, GITHUB_TEST_FRACTION, SEED)
    official = {
        "github": (gh_train, gh_test),
        "jira": (_items(_read_csv(raw_dir / "jira/train_jira.csv", True), "jira-tr"),
                 _items(_read_csv(raw_dir / "jira/test_jira.csv", True), "jira-te")),
        "so": (_items(_read_csv(raw_dir / "so/train3098itemPOLARITY.csv", False), "so-tr"),
               _items(_read_csv(raw_dir / "so/test1326itemPOLARITY.csv", False), "so-te")),
    }
    manifest = {"seed": SEED, "dev_fraction": DEV_FRACTION, "datasets": {}}
    for name, (train, test) in official.items():
        fit, dev = stratified_split(train, DEV_FRACTION, SEED)
        splits = {"fit": fit, "dev": dev, "test": test}
        ds_dir = out_dir / name
        ds_dir.mkdir(parents=True, exist_ok=True)
        for split, items in splits.items():
            write_jsonl(ds_dir / f"{split}.jsonl", items)
        orders = {split: stratified_order(splits[split], SEED) for split in ("dev", "test")}
        (ds_dir / "order.json").write_text(json.dumps(orders), encoding="utf-8")
        train_texts = {it["text"].strip().lower() for it in train}
        manifest["datasets"][name] = {
            "splits": {s: {"n": len(v), "classes": class_counts(v)} for s, v in
                       {"train": train, **splits}.items()},
            "test_texts_also_in_train": sum(it["text"].strip().lower() in train_texts for it in test),
        }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def class_counts(items: list[dict]) -> dict:
    c = Counter(it["label"] for it in items)
    return {label: c[label] for label in LABELS}


def write_jsonl(path: Path, items: list[dict]) -> None:
    path.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_split(dataset: str, split: str, out_dir: Path = PROCESSED_DIR) -> list[dict]:
    """split: fit, dev, test, or train (= fit + dev)."""
    if split == "train":
        return load_split(dataset, "fit", out_dir) + load_split(dataset, "dev", out_dir)
    return read_jsonl(out_dir / dataset / f"{split}.jsonl")


def load_sample(dataset: str, split: str, n: int | None, out_dir: Path = PROCESSED_DIR) -> list[dict]:
    """The first n items of the split's stratified order (all items when n is None)."""
    items = load_split(dataset, split, out_dir)
    if n is None:
        return items
    order = json.loads((out_dir / dataset / "order.json").read_text(encoding="utf-8"))[split]
    by_id = {it["id"]: it for it in items}
    return [by_id[i] for i in order[:n]]
