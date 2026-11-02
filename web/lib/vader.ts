// A line-by-line port of vaderSentiment 3.3.2 (Hutto & Gilbert 2014, MIT licence) for the
// browser try-box. The lexicons are exported from the Python package by `sentibench report`
// (public/data/models/vader.json). scripts/parity.ts checks this port against the Python
// compound scores on every test item; quirks of the original (such as the `but` rule
// re-finding the first equal score) are kept on purpose so the two agree.

export type VaderData = {
  lexicon: Record<string, number>;
  emojis: Record<string, string>;
  booster: Record<string, number>;
  negate: string[];
  special: Record<string, number>;
};

const C_INCR = 0.733;
const N_SCALAR = -0.74;
const PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
// Python's str.split() whitespace set, built from code points so no raw separator characters sit in the source.
const WS_CHARS = [0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x85, 0xa0, 0x1680,
  0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a,
  0x2028, 0x2029, 0x202f, 0x205f, 0x3000].map((c) => "\\u" + c.toString(16).padStart(4, "0")).join("");
const WHITESPACE = new RegExp(`[${WS_CHARS}]+`);
const EDGE_WHITESPACE = new RegExp(`^[${WS_CHARS}]+|[${WS_CHARS}]+$`, "g");

function isUpper(s: string): boolean {
  // Python: every cased character is upper case and there is at least one.
  return s === s.toUpperCase() && s !== s.toLowerCase();
}

function codePoints(s: string): number {
  return Array.from(s).length;
}

function stripPunctuation(token: string): string {
  let a = 0;
  let b = token.length;
  while (a < b && PUNCTUATION.includes(token[a])) a++;
  while (b > a && PUNCTUATION.includes(token[b - 1])) b--;
  const stripped = token.slice(a, b);
  // Two or fewer characters left: probably an emoticon such as ":)"; keep the original.
  return codePoints(stripped) <= 2 ? token : stripped;
}

// Python's round(x, 4) for the values VADER produces (no exact binary ties occur in practice).
function round4(x: number): number {
  return Number(x.toFixed(4));
}

export class Vader {
  private lexicon: Map<string, number>;
  private emojis: Map<string, string>;
  private booster: Map<string, number>;
  private negate: Set<string>;
  private special: Map<string, number>;

  constructor(data: VaderData) {
    this.lexicon = new Map(Object.entries(data.lexicon));
    this.emojis = new Map(Object.entries(data.emojis));
    this.booster = new Map(Object.entries(data.booster));
    this.negate = new Set(data.negate);
    this.special = new Map(Object.entries(data.special));
  }

  private negated(word: string): boolean {
    return this.negate.has(word) || word.includes("n't");
  }

  private scalarIncDec(word: string, valence: number, isCapDiff: boolean): number {
    const lower = word.toLowerCase();
    let scalar = 0.0;
    const boost = this.booster.get(lower);
    if (boost !== undefined) {
      scalar = boost;
      if (valence < 0) scalar *= -1;
      if (isUpper(word) && isCapDiff) {
        if (valence > 0) scalar += C_INCR;
        else scalar -= C_INCR;
      }
    }
    return scalar;
  }

  compound(input: string): number {
    // Replace emojis with their text descriptions.
    let text = "";
    let prevSpace = true;
    for (const ch of input) {
      const description = this.emojis.get(ch);
      if (description !== undefined) {
        if (!prevSpace) text += " ";
        text += description;
        prevSpace = false;
      } else {
        text += ch;
        prevSpace = ch === " ";
      }
    }
    text = text.replace(EDGE_WHITESPACE, "");

    const words = text.split(WHITESPACE).filter((w) => w.length > 0).map(stripPunctuation);
    const lower = words.map((w) => w.toLowerCase());
    const allCaps = words.filter(isUpper).length;
    const capDifferential = words.length - allCaps;
    const isCapDiff = capDifferential > 0 && capDifferential < words.length;

    let sentiments: number[] = [];
    for (let i = 0; i < words.length; i++) {
      const item = words[i];
      if (this.booster.has(lower[i])) {
        sentiments.push(0);
        continue;
      }
      if (i < words.length - 1 && lower[i] === "kind" && lower[i + 1] === "of") {
        sentiments.push(0);
        continue;
      }
      sentiments.push(this.sentimentValence(item, i, words, lower, isCapDiff));
    }
    sentiments = this.butCheck(lower, sentiments);
    return this.scoreValence(sentiments, text);
  }

  private sentimentValence(item: string, i: number, words: string[], lower: string[], isCapDiff: boolean): number {
    const itemLower = lower[i];
    let valence = 0;
    const base = this.lexicon.get(itemLower);
    if (base === undefined) return valence;
    valence = base;
    // "no" before another lexicon word negates it instead of counting itself.
    if (itemLower === "no" && i !== words.length - 1 && this.lexicon.has(lower[i + 1])) {
      valence = 0.0;
    }
    if ((i > 0 && lower[i - 1] === "no") || (i > 1 && lower[i - 2] === "no") ||
        (i > 2 && lower[i - 3] === "no" && (lower[i - 1] === "or" || lower[i - 1] === "nor"))) {
      valence = base * N_SCALAR;
    }
    if (isUpper(item) && isCapDiff) {
      if (valence > 0) valence += C_INCR;
      else valence -= C_INCR;
    }
    for (let startI = 0; startI < 3; startI++) {
      if (i > startI && !this.lexicon.has(lower[i - (startI + 1)])) {
        let s = this.scalarIncDec(words[i - (startI + 1)], valence, isCapDiff);
        if (startI === 1 && s !== 0) s = s * 0.95;
        if (startI === 2 && s !== 0) s = s * 0.9;
        valence = valence + s;
        valence = this.negationCheck(valence, lower, startI, i);
        if (startI === 2) valence = this.specialIdiomsCheck(valence, lower, i);
      }
    }
    return this.leastCheck(valence, lower, i);
  }

  private negationCheck(valence: number, w: string[], startI: number, i: number): number {
    if (startI === 0) {
      if (this.negated(w[i - 1])) valence = valence * N_SCALAR;
    }
    if (startI === 1) {
      if (w[i - 2] === "never" && (w[i - 1] === "so" || w[i - 1] === "this")) {
        valence = valence * 1.25;
      } else if (w[i - 2] === "without" && w[i - 1] === "doubt") {
        // unchanged
      } else if (this.negated(w[i - 2])) {
        valence = valence * N_SCALAR;
      }
    }
    if (startI === 2) {
      if ((w[i - 3] === "never" && (w[i - 2] === "so" || w[i - 2] === "this")) ||
          (w[i - 1] === "so" || w[i - 1] === "this")) {
        valence = valence * 1.25;
      } else if (w[i - 3] === "without" && (w[i - 2] === "doubt" || w[i - 1] === "doubt")) {
        // unchanged
      } else if (this.negated(w[i - 3])) {
        valence = valence * N_SCALAR;
      }
    }
    return valence;
  }

  private specialIdiomsCheck(valence: number, w: string[], i: number): number {
    const onezero = `${w[i - 1]} ${w[i]}`;
    const twoonezero = `${w[i - 2]} ${w[i - 1]} ${w[i]}`;
    const twoone = `${w[i - 2]} ${w[i - 1]}`;
    const threetwoone = `${w[i - 3]} ${w[i - 2]} ${w[i - 1]}`;
    const threetwo = `${w[i - 3]} ${w[i - 2]}`;
    for (const seq of [onezero, twoonezero, twoone, threetwoone, threetwo]) {
      const v = this.special.get(seq);
      if (v !== undefined) {
        valence = v;
        break;
      }
    }
    if (w.length - 1 > i) {
      const v = this.special.get(`${w[i]} ${w[i + 1]}`);
      if (v !== undefined) valence = v;
    }
    if (w.length - 1 > i + 1) {
      const v = this.special.get(`${w[i]} ${w[i + 1]} ${w[i + 2]}`);
      if (v !== undefined) valence = v;
    }
    for (const ngram of [threetwoone, threetwo, twoone]) {
      const b = this.booster.get(ngram);
      if (b !== undefined) valence = valence + b;
    }
    return valence;
  }

  private leastCheck(valence: number, w: string[], i: number): number {
    if (i > 1 && !this.lexicon.has(w[i - 1]) && w[i - 1] === "least") {
      if (w[i - 2] !== "at" && w[i - 2] !== "very") valence = valence * N_SCALAR;
    } else if (i > 0 && !this.lexicon.has(w[i - 1]) && w[i - 1] === "least") {
      valence = valence * N_SCALAR;
    }
    return valence;
  }

  private butCheck(lower: string[], sentiments: number[]): number[] {
    const bi = lower.indexOf("but");
    if (bi === -1) return sentiments;
    for (let k = 0; k < sentiments.length; k++) {
      const sentiment = sentiments[k];
      const si = sentiments.indexOf(sentiment); // first equal score, as in the original
      if (si < bi) sentiments[si] = sentiment * 0.5;
      else if (si > bi) sentiments[si] = sentiment * 1.5;
    }
    return sentiments;
  }

  private scoreValence(sentiments: number[], text: string): number {
    if (sentiments.length === 0) return 0.0;
    let sum = 0;
    for (const s of sentiments) sum += s;
    const ep = Math.min((text.match(/!/g) || []).length, 4) * 0.292;
    const qmCount = (text.match(/\?/g) || []).length;
    const qm = qmCount > 1 ? (qmCount <= 3 ? qmCount * 0.18 : 0.96) : 0;
    const emphasis = ep + qm;
    if (sum > 0) sum += emphasis;
    else if (sum < 0) sum -= emphasis;
    let compound = sum / Math.sqrt(sum * sum + 15);
    compound = Math.max(-1, Math.min(1, compound));
    return round4(compound);
  }
}

export function vaderLabel(compound: number, thresholds: [number, number] = [-0.05, 0.05]): string {
  if (compound >= thresholds[1]) return "positive";
  if (compound <= thresholds[0]) return "negative";
  return "neutral";
}
